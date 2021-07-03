import torch


class TorchBackend:
    name = 'torch'

    @classmethod
    def input_checks(cls, x):
        if x is None:
            raise TypeError('The input should be not empty.')

        cls.contiguous_check(x)

    @classmethod
    def complex_check(cls, x):
        if not cls._is_complex(x):
            raise TypeError('The input should be complex (got %s).' % x.dtype)

    @classmethod
    def real_check(cls, x):
        if not cls._is_real(x):
            raise TypeError('The input should be real (got %s).' % x.dtype)

    @classmethod
    def complex_contiguous_check(cls, x):
        cls.complex_check(x)
        cls.contiguous_check(x)

    @staticmethod
    def contiguous_check(x):
        if not x.is_contiguous():
            raise RuntimeError('Tensors must be contiguous.')

    @staticmethod
    def _is_complex(x):
        return torch.is_complex(x)

    @staticmethod
    def _is_real(x):
        return 'float' in str(x.dtype)

    @classmethod
    def modulus(cls, x, out=None):
        if out is not None:
            return torch.abs(x, out=out)
        cls.complex_check(x)
        return torch.abs(x)

    @staticmethod
    def concatenate(arrays, axis=-2):
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def concatenate_v2(arrays, axis=2):
        return torch.cat(arrays, dim=axis)

    @classmethod
    def cdgmm(cls, A, B, out=None):
        """Complex pointwise multiplication.

            Complex pointwise multiplication between (batched) tensor A and tensor B.

            Parameters
            ----------
            A : tensor
                A is a complex tensor of size (B, C, M, N, 2).
            B : tensor
                B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1).
            inplace : boolean, optional
                If set to True, all the operations are performed in place.

            Raises
            ------
            RuntimeError
                In the event that the filter B is not a 3-tensor with a last
                dimension of size 1 or 2, or A and B are not compatible for
                multiplication.

            TypeError
                In the event that A is not complex, or B does not have a final
                dimension of 1 or 2, or A and B are not of the same dtype, or if
                A and B are not on the same device.

            Returns
            -------
            C : tensor
                Output tensor of size (B, C, M, N, 2) such that:
                C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

        """
        if out is not None:
            torch.multiply(A, B, out=out)
            return out
        if not cls._is_real(B):
            cls.complex_check(B)
        cls.complex_check(A)

        sa, sb = A.shape, B.shape
        # last dims equal, or last except *the* last if the last is 1 in A or B
        if not ((sa[-B.ndim:] == sb) or
                ((sa[-1] == 1 or sb[-1] == 1) and (sa[-B.ndim:-1] == sb[:-1]))):
            raise RuntimeError('The inputs are not compatible for '
                               'multiplication (%s and %s).' % (sa, sb))

        # if A.dtype is not B.dtype:
        # TODO does "last dim 2" allow faster multiplication? (real * complex)
        #     raise TypeError('Input and filter must be of the same dtype.')

        if B.device.type == 'cuda':
            if A.device.type == 'cuda':
                if A.device.index != B.device.index:
                    raise TypeError('Input and filter must be on the same GPU.')
            else:
                raise TypeError('Input must be on GPU.')

        if B.device.type == 'cpu':
            if A.device.type == 'cuda':
                raise TypeError('Input must be on CPU.')

        return A * B

    @classmethod
    def sqrt(cls, x):
        if isinstance(x, (float, int)):
            x = torch.tensor(x)
        return torch.sqrt(x)

    @classmethod
    def conj(cls, x, inplace=False):
        if inplace:
            raise Exception("Torch autograd doesn't support `out=`")
        return (torch.conj(x) if cls._is_complex(x) else
                x)
