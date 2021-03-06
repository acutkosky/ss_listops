import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# copied from https://github.com/ag1988/dss/blob/main/src/models/sequence/ss/standalone/dss.py#L199
def hippo_skew_evals(N):
    """ eigenvalues of (Hippo - Hippo.t()) / 2  (largest imag part first) """
    i = torch.arange(N, dtype=torch.float)
    x = 2*i + 1
    Hippo = (x.view(-1,1) * x.view(1,-1)).sqrt().tril(diagonal=-1)  # [N N]
    Skew = (Hippo - Hippo.t()) / 2                                  # [N N]
    evals = torch.linalg.eigvals(Skew)                              # [N]
    # decreasing order of imag
    return evals[evals.imag.argsort(descending=True)]               # [N]


def get_propogator(frequencies, decays):
    exponents = torch.complex(-torch.exp(decays), frequencies)
    return torch.exp(exponents)

def get_kernel(frequencies, decays, length):
    exponents = torch.complex(-torch.exp(decays), frequencies)

    exponents = exponents.unsqueeze(0).tile((length, 1)) # [N] -> [L, N]

    exponents = exponents * torch.arange(length).unsqueeze(1).to(exponents.device) # [L, N] * [L, 1] -> [L, N]

    exponents = exponents.exp()
    #exponents = exponents/torch.sum(exponents)

    return exponents


class SimpleState(nn.Module):
    def __init__(self,
                 d_model,
                 d_state,
                 real_init=-0.5,
                 scaling=1,
                 bidirectional=False,
                 transposed=False,
                 **random_other_args # put this in for compatibility with dss code, probably shouldn't have it
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed
        self.bidirectional = bidirectional

        if self.bidirectional:
            assert d_state % 2 == 0, "need even d_state for bidirectional to divide into two directions!"


        self.default_initial = nn.Parameter(torch.randn(1, 1, d_state))
        self.in_projection = nn.Parameter(torch.randn(d_model, d_state))
        self.out_projection = nn.Parameter(torch.randn(d_state, d_model))

        self.frequencies = hippo_skew_evals(2*d_state)[:d_state].imag

        if self.bidirectional:
            # we rearrange the frequencies here to avoid having to rearrange in every
            # forward pass. This rearrangement roughly even allocates initial frequency
            # magnitudes to both forward and backward propogation.
            f_0, f_1 = rearrange(self.frequencies, '(n s) -> s n', s=2)
            self.frequencies = torch.concat((f_0, f_1))

        self.frequencies = nn.Parameter(self.frequencies.detach().float() * scaling)
        self.decays = nn.Parameter(torch.full_like(self.frequencies, np.log(np.abs(real_init) * scaling +1e-10)).float())




    def forward(self, input, initial_state=None):
        '''
        input: B H L if tranposed, B L H otherwise, where B=batch, H=hidden size (i.e. embedding dim)
            and L = sequence length
        initial_state: [B, N]. Starting value for the state (e.g. from
            previous block). If set to None, self.default_initial is used
            (this is probably fine in most cases).
            N = size of state representation.

        returns output (same shape as input) and final state [B, N]

        Note that final state may not be meaningful if bidirectional=True.
        In fact, for most scenarios the "final state" can be ignored. It is possibly
        useful for some streaming type situation though, so we provide it here.


        '''

        if self.transposed:
            input = input.transpose(-2, -1)

        # input is now B L H
        B, L, H = input.size()


        if initial_state is None:
            initial_state = self.default_initial.tile((B, 1, 1)) # [B, 1, N]
        else:
            initial_state = initial_state.unsqueeze(1) # [B, 1, N]

        u = input @ self.in_projection # [B L H] -> [B L N], where N is d_state

        u = torch.cat((initial_state, u), 1) # [B, L+1, H]
        kernel_len = L + 1
        if self.bidirectional:
            u = torch.cat((u, initial_state), 1) # [B, L+2, H]
            kernel_len = L + 2

        fft_len = 2 * kernel_len - 1

        kernel = get_kernel(self.frequencies, self.decays, kernel_len) # [K, N] (complex) K=L+1 or L+2

        if self.bidirectional:
            # this is a bit tricky: we don't actually want to simply reverse
            # the kernel to propogate backwards.
            # Instead, we want to reverse everything *except the first entry*.
            # for example, suppose the kernel is [1, z, z^2, z^3]
            # and the input is [a, b, c ,d]
            # Then for backwards propogation the first entry should be:
            # a + b z + c z^2 + d z^3
            # if you flip the kernel and pad and convolve the flipped+padded kernel
            # is [0, 0, 0, z^3, z^2, z, 1]
            # which yields first entry b + c z + d z^2
            # instead, we want to convolve with [1, 0, 0, 0, z^2, z^2, z].
            # We could also accomplish this by padding and then taking conjugate in fft space if
            # the kernel is a real kernel (although I think in this implementation currently it is not...)
            # but flipping in signal space is more intuitive. TODO: investigate if an fft-based
            # method is faster.

            # Note the ordering (s n) rather than (n s) in the rearrange below important!
            # in __init__ we rearranged the frequencies so that the first and second
            # halves of the frequency list have approximately equally distributed magnitudes.
            k_forward, k_backward_flip = rearrange(kernel, 'k (s n) -> s k n', s=2) # 2 tensors both [K, N/2]
            k_forward_pad = F.pad(k_forward, (0, 0, 0, fft_len-kernel_len)) #[fft_len, N/2]
            k_backward_pad_flip = F.pad(k_backward_flip, (0, 0, 0, fft_len-kernel_len)) #[fft_len, N/2]

            k_backward_pad = torch.concat((k_backward_pad_flip[0].unsqueeze(0), k_backward_pad_flip[1:, :].flip(0)))
            kernel = torch.concat((k_forward_pad, k_backward_pad), dim=1) #[fft_len, N]


        # we use full fft rather than real fft since we don't constrain our
        # frequencies to be the eigenvalues of a real matrix...
        # we could instead take the real part of the kernel first,
        # but this seems more natural to me.
        u_f = torch.fft.fft(u, n=fft_len, dim=-2)
        kernel_f = torch.fft.fft(kernel, n=fft_len, dim=-2)

        conv_f = u_f * kernel_f

        conv = torch.fft.ifft(conv_f, n=fft_len, dim=-2)[:, 1:L+1, :]  # [B, L, N] (real)

        final_state = conv[:, -1, :].unsqueeze(1) # [B, 1, N] -> [B, N]

        output = conv.real @ self.out_projection     # [B, L, H]

        if self.transposed:
            output = output.transpose(-2, -1)  #[B, L, H] -> [B, H, L]

        return output, final_state


    def propogate(self, input, initial_state=None):
        '''
        input: [B, H, L] if transposed, [B, L, H] otherwise
        initial_state: initial state, [B, N]

        returns
        state sequence (same shape as input)
        final state [B, N]

        propogates manually without fft. Should be identical result to
        forward. This is much slower during training, but could (maybe?)
        be used for faster inference...
        '''


        if self.transposed:
            input = input.transpose(-2, -1)

        # input is now B L H
        B, L, H = input.size()


        if initial_state is None:
            initial_state = self.default_initial.tile((B, 1, 1)) # [B, 1, N]
        else:
            initial_state = initial_state.unsqueeze(1) # [B, 1, N]

        u = input @ self.in_projection # [B L H] -> [B L N], where N is d_states

        propogator = get_propogator(self.frequencies, self.decays) # [N]

        def propogate_one_direction(propogator, initial_state, u):
            # I'm sure there is a "right" way to do this, but this is what we're
            # doing right now because I never really got the hand of not
            # using for loops...
            current_state = initial_state
            output = []
            for idx in range(L):
                current_state = current_state * propogator + u[:, idx, :].unsqueeze(1)
                output.append(current_state.real)

            output = torch.concat(output, dim=1) # list ([B, 1, N]) -> [B, L, N]
            return output, current_state

        if not self.bidirectional:
            output, current_state = propogate_one_direction(propogator, initial_state, u)
        else:
            # This part is a bit annoying: in the fft-based function,
            # we divide the propogator into forward and backward frequencies
            # by taking every other frequency.
            # However, the states are divided by
            propogator_forward = propogator[:self.d_state//2]  # [N/2]
            propogator_backward = propogator[self.d_state//2:] # [N/2]
            u_forward = u[:, :, :self.d_state//2]              # [B, L, N/2]
            u_backward = u[:, :, self.d_state//2:]             # [B, L, N/2]
            u_backward_flip = u_backward.flip(-1)              # [B, L, N/2]

            initial_state_forward = initial_state[:, :, :self.d_state//2] # [1, 1, N/2]

            initial_state_backward = initial_state[:, :, self.d_state//2:] # [1, 1, N/2]

            output_forward, state_forward = propogate_one_direction(propogator_forward,
                                                                    initial_state_forward,
                                                                    u_forward)  # [B, L, N/2],  [1, 1, N/2]

            output_backward_flip, _ = propogate_one_direction(propogator_backward,
                                                                    initial_state_backward,
                                                                    u_backward_flip) # [B, L, N/2],  [1, 1, N/2]

            output_backward = output_backward_flip.flip(-2)  # [B, L, N/2]

            output = torch.concat((output_forward, output_backward), -1)  # [B, L, N]

            current_state = torch.concat((state_forward, initial_state_backward * propogator_backward), -1) # [1, 1, N]



        output = output @ self.out_projection  # [B, L, N] @ [N, H] -> [B, L, H]


        if self.transposed:
            output = output.transpose(-2, -1)  # [B, L, H] -> [B, H, L]

        return output, current_state







def _paddedfft(input, kernel_size, padding='same'):
    '''produces fft of input.
    args:
        input: the tensor to fft.
        kernel_size: the size of the kernel we will be convolving with.
        padding: the type of padding to use.
    returns:
        input_f: the real fourier transform of the input, padded appropriately
        N: the order of the fourier transform (as this is not immediately visible
            from a real fourier transform, unlike the complex transform).
        padding: the padding added to both sides of input'''
    L = input.size(-1)

    # We will assume that the 'same' padding mode is desired for now, and
    # that we pad with zeros. Circular padding is actually easier...
    # This requires us to pad the beginning of the input with roughly
    # half the kernel size of zeros.
    # technically, we don't need to pad the endsince the fft function will
    # auto-pad the end for us using the parameter N, but we'll do it anyway.
    # TODO: find out if padding is an O(N) operation that copies the
    # input (seems likely). If so, can we avoid doing it?
    if padding == 'same':
        padding = [(kernel_size - 1) // 2, kernel_size - 1 - (kernel_size - 1) // 2]
    elif padding == 'valid':
        padding = [0, 0]
    elif padding == 'postfix':
        padding = [0, kernel_size - 1]
    elif padding == 'prefix':
        padding = [kernel_size - 1, 0]

    if not hasattr(padding, '__getitem__'):
        # if padding is a single number, expand to symmetric padding.
        padding = [padding, padding]
    input_pad = F.pad(input, padding)

    # if the padding is larger than the kernel size, we need to expand the
    # order of the FFT.
    N = L + max(kernel_size - 1, padding[0] + padding[1])

    # input_pad = F.pad(input, [(kernel_size-1) // 2 ,0])
    input_f = torch.fft.rfft(input_pad, N)

    return input_f, N, padding


def fftconv1d(input, weight, bias=None, padding='same'):
    L = input.size(-1)
    kernel_size = weight.size(-1)

    input_f, N, padding = _paddedfft(input, kernel_size, padding)


    # Annoyingly, convolution layers actually compute a cross-correlation
    # rather than an actual convolution. Cross-correlation is convolution
    # with a time-reversed kernel.
    # Here 'time-reversed' is NOT the same as simply writing the kernel
    # backwards: instead, the first coordinate is kept the same but the
    # rest of the kernel is reversed. Fortunately, the fourier transform of
    # a reversed kernel is the reverse of the fourier transform of the original
    # kernel. For a real-valued kernel, it turns out that the reversed fourier
    # transform is just the complex conjugate of the original transform.

    kernel_f = torch.fft.rfft(weight, N)
    # time-reverse the kernel for cross-correlation:
    rev_kernel_f = torch.conj(kernel_f)


    conv_f = torch.einsum('oi...,bi...->bo...', rev_kernel_f, input_f)


    output = torch.fft.irfft(conv_f, N)

    # now we select the parts of the convolution that we actually want.
    # If no padding is desired, we should change start and end.
    # If a stride of s is desired, we should only pick every s^th element
    # of output
    start = 0

    # following code is unnecessary if we pad both the end and the beginning
    # of input.
    # TODO: find out if it's faster not to pad the end of input.
    # if padding == 'same':
    #     end = L
    # elif padding == 'valid':
    #     end = L - kernel_size + 1
    # elif padding == 'postfix':
    #     end = L
    # elif padding == 'prefix':
    #     end = L
    # else:
    end = L - kernel_size + 1 + padding[0] + padding[1]


    output = output[:, :, start:end]

    if bias is not None:
        output = output + bias.unsqueeze(-1)

    return output


class FFTConv1d(nn.Module):
    '''Conv1d using FFT and convolution theorem.
    Currently only supports inputs of shape [B, C, L], stride 1, no padding
    or dilation.
    Should be significantly faster than pytroch conv1d for very large kernel
    sizes.'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
        device=None, dtype=None, init='normal', from_conv=None):
        super(FFTConv1d, self).__init__()

        assert init in ['normal', 'zero']

        if from_conv is None:
            if init == 'normal':
                k = groups/(in_channels * kernel_size)
            else:
                k = 0.0

            # this initialization scheme is copied from pytorch's conv1d implementation
            self.weight = nn.Parameter(np.sqrt(k) * torch.randn(out_channels, in_channels, kernel_size))
            if device is not None:
                self.weight.to(device)
            self.bias = None
            if bias:
                self.bias = nn.Parameter(np.sqrt(k) * torch.randn(out_channels))
                if device is not None:
                    self.bias.to(device)
        else:
            self.weight = from_conv.weight
            self.bias = from_conv.bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.gorups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        # TODO: correctly handle alternative settings of these parameters
        assert stride == 1
        assert padding_mode == 'zeros'
        assert dilation == 1
        assert groups == 1


    def forward(self, input):
        return fftconv1d(input, self.weight, self.bias, self.padding)

def multi_way_correlation(*args, normalize=True):
    '''experimental correlation function. Correlates (or "convolves" in DL lingo)
    in several different signals.'''

    assert len(args) > 1

    input_size = args[0].size(-1)


    if normalize:
        normalizer = 1.0/torch.sqrt(torch.arange(1, input_size+1))
    else:
        normalizer = torch.ones(input_size)

    accumulator = args[0]
    for input in args[1:]:
        input_f, N, _ = _paddedfft(input, input_size, padding='prefix')
        accumulator_f = torch.fft.rfft(accumulator, N)
        accumulator_f = torch.conj(accumulator_f)
        accumulator = accumulator_f*input_f

        # TODO: do we really need to jump back and forth between
        # frequency and signal domain here?
        accumulator = torch.fft.irfft(accumulator, N)[: ,:, 0:input_size]
        accumulator *= normalizer

    return accumulator




class DiagAutoCorrelation(nn.Module):
    '''experimental "triple" autocorrelation module.
    I suspect the most interesting "order" is 3: ignoring the extra convolutions,
    this corresponds to convolving the input with its autocorrelation function.
    Intuitively, this may "clean up" any periodic components in the input signal.'''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, order=3, init='normal'):
        super(DiagAutoCorrelation, self).__init__()

        self.conv_list = [FFTConv1d(in_channels, out_channels, kernel_size, bias, padding='prefix', init=init) for _ in range(order)]

    def forward(self, input):

        input_convs_list = [conv(input) for conv in self.conv_list]

        return multi_way_correlation(*input_convs_list)


def test_simplestate():

    B = 10
    H = 15
    N = 20
    L = 30

    ss = SimpleState(H, N)
    x = torch.randn(B, H, L)


    # these have radically ifferent enough implementations that I'd not expect
    # them to agree unless both were correct...
    fft_forward, fft_state = ss(x)
    manual_forward, manual_state = ss.propogate(x)


    assert torch.allclose(fft_forward, manual_forward, atol=1e-4)
    assert torch.allclose(fft_state, manual_state, atol=1e-4)


    ss_bd = SimpleState(H, N, bidirectional=True)

    fft_fwd_bd, fft_state_bd = ss_bd(x)

    prop_fwd_bd, prop_state_bd  = ss_bd.propogate(x)

    assert torch.allclose(fft_fwd_bd, prop_fwd_bd, atol=1e-4)
    assert torch.allclose(fft_state_bd, prop_state_bd, atol=1e-4)

# a simple test
if __name__=='__main__':


    test_simplestate()

    B = 3
    C = 2
    L = 8

    O = 20
    K = 7
    x = torch.randn(B,C,L)

    def test(padding):
        conv = FFTConv1d(C, O, K, bias=True, padding=padding)

        weight = conv.weight
        bias = conv.bias

        our_conv = conv(x)

        tconv = torch.nn.Conv1d(C, O, K, bias=True, padding=padding)
        tconv.weight = weight
        tconv.bias = bias
        torch_conv = tconv(x)



        # make sure autograd doesn't throw an error...
        assert conv.weight.grad is None
        assert conv.bias.grad is None

        torch.sum(our_conv).backward()

        assert conv.weight.grad is not None
        assert conv.bias.grad is not None


        assert torch.allclose(our_conv,torch_conv, atol=1e-6)


    test(padding='same')
    test(padding='valid')
    test(padding=1)
    test(padding=5)

    # unfortunately I cannot compare the more general padding = [a,b] tuple
    # with pytorch's implementation since pytorch's conv1d doesn't actually
    # support this, even though their docs suggest that it should...
    # So, instead, we will just test a small example that can be manually
    # verified.

    x = torch.tensor([[[1,2,3,4]]]).float()
    k = torch.tensor([[[1,2]]]).float()
    b = torch.tensor([1]).float()

    padded_conv = FFTConv1d(1, 1, 2, padding=[2,1])
    padded_conv.weight = torch.nn.Parameter(k)
    padded_conv.bias = torch.nn.Parameter(b)

    answer = padded_conv(x)
    expected_answer = torch.tensor([[[1, 3, 6, 9, 12, 5]]]).float()
    assert torch.allclose(answer, expected_answer, atol=1e-6)


    # next, let us test the 'prefix' padding mode, which are not
    # available in pytorch. 'prefix' padds only the beginning of input
    # so as to produce an output of the same length (unlike 'same', which
    # attempts to pad both beginning and end of input roughly the same
    # amount).

    padded_conv = padded_conv = FFTConv1d(1, 1, 2, padding='prefix')
    padded_conv.weight = torch.nn.Parameter(k)
    padded_conv.bias = torch.nn.Parameter(b)
    answer = padded_conv(x)
    expected_answer = torch.tensor([[[3, 6, 9, 12]]]).float()
    assert torch.allclose(answer, expected_answer, atol=1e-6)

    # finally, we test 'postfix' mode, which pads the end of input:

    padded_conv = padded_conv = FFTConv1d(1, 1, 2, padding='postfix')
    padded_conv.weight = torch.nn.Parameter(k)
    padded_conv.bias = torch.nn.Parameter(b)
    answer = padded_conv(x)
    expected_answer = torch.tensor([[[6, 9, 12, 5]]]).float()
    assert torch.allclose(answer, expected_answer, atol=1e-6)



    # test that the autocorellation module at least doesn't throw an error
    # and returns the right output shape.
    # TODO make this test more detailed
    autocor = DiagAutoCorrelation(C, O, K, order=5)
    x = torch.randn(B, C, L)
    answer = autocor(x)
    assert answer.shape == torch.Size([B, O, L])


    # test multi way correlation module on specific example

    x = torch.tensor([[[1,0,1,0,1,0]]]).float()
    answer = multi_way_correlation(x,x,x, normalize=False)
    expected_answer = torch.tensor([[[3, 0, 5, 0, 6, 0]]]).float()
    assert torch.allclose(answer, expected_answer, atol=1e-6)


    print("It's working!")
