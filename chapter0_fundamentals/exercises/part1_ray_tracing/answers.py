# %%
import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros(num_pixels, 2, 3)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = t.linspace(-y_limit, y_limit, steps=num_pixels)
    return rays


rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)

# %%

if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)


# %%

segments = t.tensor(
    [
        [[1.0, -12.0, 0.0], [1, -6.0, 0.0]],
        [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]],
        [[2, 12.0, 0.0], [2, 21.0, 0.0]],
    ]
)

if MAIN:
    render_lines_with_plotly(rays1d, segments)


# %%
@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(ray, segment) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """

    O, D = ray[:, :2]
    L1, L2 = segment[:, :2]
    A = t.stack([D, L1 - L2], dim=1)

    try:
        x = t.linalg.solve(A, L1 - O)
    except t.linalg.LinAlgError:
        return False

    u = x[0].item()
    v = x[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    nrays, nsegments = rays.shape[0], segments.shape[0]

    O, D = rays[:, 0, :2], rays[:, 1, :2]
    O = einops.repeat(O, "r d -> (r s) d", s=nsegments)
    D = einops.repeat(D, "r d -> (r s) d", s=nsegments)

    L1, L2 = segments[:, 0, :2], segments[:, 1, :2]
    L1 = einops.repeat(L1, "s d -> (r s) d", r=nrays)
    L2 = einops.repeat(L2, "s d -> (r s) d", r=nrays)

    A = t.stack([D, L1 - L2], dim=-1)
    mask = t.linalg.det(A).abs() < 1e-8
    A[mask] = t.eye(2)
    X = t.linalg.solve(A, L1 - O)
    us = X[:, 0]
    vs = X[:, 1]

    result = (~mask) & (0 <= us) & (0 <= vs) & (vs <= 1)

    return einops.rearrange(result, "(r s)-> r s", r=nrays).any(dim=1)


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%


def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[t.Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """

    rays = t.zeros(num_pixels_y * num_pixels_z, 2, 3)
    rays[:, 1, 0] = 1

    y_span = t.linspace(-y_limit, y_limit, steps=num_pixels_y)
    rays[:, 1, 1] = einops.repeat(y_span, "y -> (y z)", z=num_pixels_z)

    z_span = t.linspace(-z_limit, z_limit, steps=num_pixels_y)
    rays[:, 1, 2] = einops.repeat(z_span, "z -> (y z)", y=num_pixels_y)

    return rays


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)

# %%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)


if MAIN:
    display(fig)

# %%

Point = Float[Tensor, "points=3"]


@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """

    mat = t.stack([-D, B - A, C - A], dim=1)
    b = O - A

    try:
        _, u, v = t.linalg.solve(mat, b)
        u, v = u.item(), v.item()
    except t.linalg.LinAlgError:
        return False

    return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%


def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.shape[0]
    O, D = rays.unbind(1)
    A, B, C = einops.repeat(triangle, "p d -> p r d", r=nrays)

    mat = t.stack([D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    _, us, vs = t.linalg.solve(mat, O - A).unbind(1)

    return (0 <= us) & (0 <= vs) & (us + vs <= 1) & ~is_singular


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 35
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%


def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    NR = rays.size()[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([-D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular


# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

# %%


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays, ntriangles = rays.shape[0], triangles.shape[0]
    O, D = einops.repeat(rays, "r p d -> p t r d", t=ntriangles)
    A, B, C = einops.repeat(triangles, "t p d -> p t r d", r=nrays)

    mat = t.stack([D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    s, u, v = t.linalg.solve(mat, O - A).unbind(2)
    intersections = (0 <= u) & (0 <= v) & (u + v <= 1) & ~is_singular
    s[~intersections] = float("inf")

    return s.min(dim=0).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(
        img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000
    )
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]):
        fig.layout.annotations[i]["text"] = text
    fig.show()

# %%


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    # SOLUTION
    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = float("inf")  # t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(
        img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000
    )
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]):
        fig.layout.annotations[i]["text"] = text
    fig.show()

# %%
