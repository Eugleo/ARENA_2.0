# %%

import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

# %%

arr = np.load(section_dir / "numbers.npy")

# %%
if MAIN:
    display_array_as_img(arr[0])

# %%

arr1 = einops.rearrange(arr, "n c h w -> c h (n w)")

if MAIN:
    display_array_as_img(arr1)
# %%

arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")

if MAIN:
    display_array_as_img(arr2)
# %%

arr3 = einops.repeat(arr[:2], "n c h w -> c (n h) (2 w)")

if MAIN:
    display_array_as_img(arr3)
# %%

arr4 = einops.rearrange(arr[0], "c (h x) w -> c h (w x)", x=2)

if MAIN:
    display_array_as_img(arr4)
# %%

arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")

if MAIN:
    display_array_as_img(arr5)

# %%


def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return einops.einsum(mat, "i i ->")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec, "i j, j -> i")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einops.einsum(mat1, mat2, "i k, k j -> i j")


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return einops.einsum(vec1, vec2, "i, i ->")


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return einops.einsum(vec1, vec2, "i, j -> i j")


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)

# %%

if MAIN:
    test_input = t.tensor(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ],
        dtype=t.float,
    )

# %%
from collections import namedtuple


if MAIN:
    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]),
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]),
            size=(2, 2),
            stride=(5, 2),
        ),
        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4,),
            stride=(5,),
        ),
        TestCase(
            output=t.tensor([[0, 1, 2], [5, 6, 7]]),
            size=(2, 3),
            stride=(5, 1),
        ),
        TestCase(
            output=t.tensor([[0, 1, 2], [10, 11, 12]]),
            size=(2, 3),
            stride=(10, 1),
        ),
        TestCase(
            output=t.tensor([[0, 0, 0], [11, 11, 11]]),
            size=(2, 3),
            stride=(11, 0),
        ),
        TestCase(
            output=t.tensor([0, 6, 12, 18]),
            size=(4,),
            stride=(6,),
        ),
    ]

    for i, test_case in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")

# %%


def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    """
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    """
    i = mat.shape[0]
    return t.sum(t.as_strided(mat, (i,), (i + 1,)))


if MAIN:
    tests.test_trace(as_strided_trace)


# %%
def as_strided_mv(
    mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]
) -> Float[Tensor, "i"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    strideV = vec.stride()
    vec_expanded = t.as_strided(vec, mat.shape, (0, strideV[0]))
    mul_expanded = mat * vec_expanded
    return mul_expanded.sum(dim=1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

# %%


def as_strided_mm(
    matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]
) -> Float[Tensor, "i k"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    i, k = matA.shape
    _, j = matB.shape

    sAi, sAk = matA.stride()
    matA_expanded = t.as_strided(matA, (i, k, j), (sAi, sAk, 0))

    sBk, sBj = matB.stride()
    matB_expanded = t.as_strided(matB, (i, k, j), (0, sBk, sBj))

    mul_expanded = matA_expanded * matB_expanded
    return mul_expanded.sum(dim=1)


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)


# %%


def conv1d_minimal_simple(
    x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]
) -> Float[Tensor, "ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    """
    w, kw = x.shape[0], weights.shape[0]
    out_width = w - kw + 1
    stride = x.stride(0)
    x_expanded = x.as_strided((out_width, kw), (stride, stride))

    return einops.einsum(x_expanded, weights, "i j, j -> i")


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)


# %%
def conv1d_minimal(
    x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]
) -> Float[Tensor, "b oc ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    b, ic, w = x.shape
    oc, _, kw = weights.shape
    ow = w - kw + 1
    s_b, s_ic, s_w = x.stride()

    x_expanded = x.as_strided(size=(b, ic, ow, kw), stride=(s_b, s_ic, s_w, s_w))

    return einops.einsum(x_expanded, weights, "b ic ow kw, oc ic kw -> b oc ow")


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)

# %%


def conv2d_minimal(
    x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    b, ic, h, w = x.shape
    _, _, kh, kw = weights.shape
    oh, ow = h - kh + 1, w - kw + 1
    s_b, s_ic, s_h, s_w = x.stride()

    x_expanded = x.as_strided(
        size=(b, ic, oh, kh, ow, kw), stride=(s_b, s_ic, s_h, s_h, s_w, s_w)
    )

    return einops.einsum(
        x_expanded, weights, "b ic oh kh ow kw, oc ic kh kw -> b oc oh ow"
    )


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)

# %%


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    b, ic, w = x.shape
    result = x.new_full((b, ic, left + right + w), pad_value)
    result[..., left : left + w] = x

    return result


if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)

# %%


def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    b, ic, h, w = x.shape
    result = x.new_full((b, ic, top + h + bottom, left + w + right), pad_value)
    result[..., top : top + h, left : left + w] = x

    return result


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)

# %%


def conv1d(
    x: Float[Tensor, "b ic w"],
    weights: Float[Tensor, "oc ic kw"],
    stride: int = 1,
    padding: int = 0,
) -> Float[Tensor, "b oc ow"]:
    """
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    x_padded = pad1d(x, left=padding, right=padding, pad_value=0)
    b, ic, w = x_padded.shape
    _, _, kw = weights.shape
    ow = 1 + (w - kw) // stride
    s_b, s_ic, s_w = x_padded.stride()

    x_expanded = x_padded.as_strided(
        size=(b, ic, ow, kw), stride=(s_b, s_ic, stride * s_w, s_w)
    )

    return einops.einsum(x_expanded, weights, "b ic ow kw, oc ic kw -> b oc ow")


if MAIN:
    tests.test_conv1d(conv1d)

# %%
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


# Examples of how this function can be used:
if MAIN:
    for v in [(1, 2), 2, (1, 2, 3)]:
        try:
            print(f"{v!r:9} -> {force_pair(v)!r}")
        except ValueError:
            print(f"{v!r:9} -> ValueError")


# %%


def conv2d(
    x: Float[Tensor, "b ic h w"],
    weights: Float[Tensor, "oc ic kh kw"],
    stride: IntOrPair = 1,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """

    padding_h, padding_w = force_pair(padding)
    stride_h, stride_w = force_pair(stride)

    x_padded = pad2d(
        x, left=padding_w, right=padding_w, bottom=padding_h, top=padding_h, pad_value=0
    )

    b, ic, h, w = x_padded.shape
    _, _, kh, kw = weights.shape
    oh, ow = (h - kh) // stride_h + 1, (w - kw) // stride_w + 1
    s_b, s_ic, s_h, s_w = x_padded.stride()

    x_expanded = x_padded.as_strided(
        size=(b, ic, oh, kh, ow, kw),
        stride=(s_b, s_ic, stride_h * s_h, s_h, stride_w * s_w, s_w),
    )

    return einops.einsum(
        x_expanded, weights, "b ic oh kh ow kw, oc ic kh kw -> b oc oh ow"
    )


if MAIN:
    tests.test_conv2d(conv2d)

# %%


def maxpool2d(
    x: Float[Tensor, "b ic h w"],
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b ic oh ow"]:
    """
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    """
    stride = kernel_size if stride is None else stride
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    x_padded = pad2d(
        x,
        left=padding_w,
        right=padding_w,
        bottom=padding_h,
        top=padding_h,
        pad_value=-t.inf,
    )

    b, ic, h, w = x_padded.shape
    kh, kw = force_pair(kernel_size)
    oh, ow = (h - kh) // stride_h + 1, (w - kw) // stride_w + 1
    s_b, s_ic, s_h, s_w = x_padded.stride()

    x_strided = x_padded.as_strided(
        size=(b, ic, oh, kh, ow, kw),
        stride=(s_b, s_ic, stride_h * s_h, s_h, stride_w * s_w, s_w),
    )

    return x_strided.amax(dim=(3, 5))


if MAIN:
    tests.test_maxpool2d(maxpool2d)

# %%


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join(
            [
                f"{key}={getattr(self, key)}"
                for key in ["kernel_size", "stride", "padding"]
            ]
        )


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

# %%


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


if MAIN:
    tests.test_relu(ReLU)

# %%

from math import prod


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape
        b = self.start_dim
        e = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        new_shape = shape[:b] + (prod(shape[b : e + 1]),) + shape[e + 1 :]

        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return ", ".join(
            [f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]]
        )


if MAIN:
    tests.test_flatten(Flatten)

# %%

from math import sqrt


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        c = 1 / sqrt(in_features)
        dist = t.distributions.Uniform(-c, c)
        self.weight = nn.Parameter(dist.sample((out_features, in_features)))
        self.bias = nn.Parameter(dist.sample((out_features,))) if bias else None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)

# %%


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()

        kh, kw = force_pair(kernel_size)
        c = sqrt(6) / sqrt(in_channels * kw * kh + out_channels + 1)
        dist = t.distributions.Uniform(-c, c)

        self.weight = nn.Parameter(dist.sample((out_channels, in_channels, kh, kw)))
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_conv2d_module(Conv2d)

# %%


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(in_features=32 * 14 * 14, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.fc(self.flatten(self.relu(self.maxpool(self.conv(x)))))


if MAIN:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    print(model)

# %%

if MAIN:
    MNIST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def get_mnist(subset: int = 1):
    """Returns MNIST training data, sampled by the frequency given in `subset`."""
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    if subset > 1:
        mnist_trainset = Subset(
            mnist_trainset, indices=range(0, len(mnist_trainset), subset)
        )
        mnist_testset = Subset(
            mnist_testset, indices=range(0, len(mnist_testset), subset)
        )

    return mnist_trainset, mnist_testset


if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=True)
# %%

if MAIN:
    img, label = mnist_trainset[1]

    imshow(
        img.squeeze(),
        color_continuous_scale="gray",
        zmin=img.min().item(),
        zmax=img.max().item(),
        title=f"Digit = {label}",
        width=450,
    )

# %%

if MAIN:
    img_input = img.unsqueeze(0).to(device)  # add batch dimension
    probs = model(img_input).squeeze().softmax(-1).detach()

    bar(
        probs,
        x=range(10),
        template="ggplot2",
        width=600,
        title="Classification probabilities",
        labels={"x": "Digit", "y": "Probability"},
        text_auto=".2f",
        showlegend=False,
        xaxis_tickmode="linear",
    )

# %%

if MAIN:
    batch_size = 64
    epochs = 3

    mnist_trainset, _ = get_mnist(subset=10)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in tqdm(range(epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(
                loss.item()
            )  # .item() converts single-elem tensor to scalar

# %%

if MAIN:
    line(
        loss_list,
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST (cross entropy loss)",
        width=700,
    )

# %%

if MAIN:
    probs = model(img_input).squeeze().softmax(-1).detach()

    bar(
        probs,
        x=range(10),
        template="ggplot2",
        width=600,
        title="Classification probabilities",
        labels={"x": "Digit", "y": "Probability"},
        text_auto=".2f",
        showlegend=False,
        xaxis_tickmode="linear",
    )

# %%
