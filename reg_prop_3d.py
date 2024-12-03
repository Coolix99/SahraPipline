#import logging
import numpy as np
import scipy.ndimage as spim
#from porespy.tools import extract_subsection, bbox_to_slices
from skimage.measure import mesh_surface_area
try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes
from skimage.morphology import skeletonize_3d, ball
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties



def regionprops_3D(im,Vox_size):
    r"""
    Calculates various metrics for each labeled region in a 3D image.

    This functions offers a few extras for 3D images that are not provided
    by the ``regionprops`` function in ``scikit-image``.

    Parameters
    ----------
    im : array_like
        An image containing at least one labeled region. If a boolean image
        is received than the ``True`` voxels are treated as a single region
        labeled ``1``.  Regions labeled 0 are ignored in all cases.

    Returns
    -------
    props : list
        An augmented version of the list returned by skimage's ``regionprops``.
        Information, such as ``volume``, can be found for region A using the
        following syntax: ``result[A-1].volume``.

        The returned list contains all the metrics normally returned by
        **skimage.measure.regionprops** plus the following:

        'slices'
            Slice indices into the image that can be used to extract the
            region

        'volume'
            Volume of the region in number of voxels.

        'bbox_volume'
            Volume of the bounding box that contains the region.

        'border'
            The edges of the region, found as the locations where the distance
            transform is 1.

        'inscribed_sphere'
            An image containing the largest sphere can can fit entirely inside
            the region.

        'surface_mesh_vertices'
            Obtained by applying the marching cubes algorithm on the region,
            AFTER first blurring the voxel image. This allows marching cubes
            more freedom to fit the surface contours.
            See also ``surface_mesh_simplices``

        'surface_mesh_simplices'
            This accompanies ``surface_mesh_vertices`` and together they can
            be used to define the region as a mesh.

        'surface_area'
            Calculated using the mesh obtained as described above, using the
            ``porespy.metrics.mesh_surface_area`` method.

        'sphericity'
            Defined as the ratio of the area of a sphere with the same volume
            as the region to the actual surface area of the region.

        'skeleton'
            The medial axis of the region obtained using the ``skeletonize_3D``
            method from **skimage**.

        'convex_volume'
            Same as convex_area, but translated to a more meaningful name.

    See Also
    --------
    snow_partitioning

    Notes
    -----
    Regions can be identified using a watershed algorithm, which can be a bit
    tricky to obtain desired results.  *PoreSpy* includes the SNOW algorithm,
    which may be helpful.



    """
    results = regionprops(im)
    for i, obj in enumerate(results):
        a = results[i]
        b = RegionPropertiesPS(a.slice,
                               a.label,
                               a._label_image,
                               a._intensity_image,
                               a._cache_active,
                               Vox_size)
        results[i] = b

    return results


class RegionPropertiesPS(RegionProperties):
    def __init__(self, slice,label,_label_image,_intensity_image,_cache_active,Vox_size):
        RegionProperties.__init__(self, slice,label,_label_image,_intensity_image,_cache_active)
        self.Vox_size = Vox_size

    @property
    def mask(self):
        return self.image

    # @property
    # def slices(self):
    #     return self._slice

    @property
    def volume(self):
        return self.area*self.Vox_size[0]*self.Vox_size[1]*self.Vox_size[2]

    # @property
    # def bbox_volume(self):
    #     mask = self.mask
    #     return np.prod(mask.shape)

    # @property
    # def border(self):
    #     return self.dt == 1

    # @property
    # def dt(self):
    #     mask = self.mask
    #     mask_padded = np.pad(mask, pad_width=1, mode='constant')
    #     temp = edt(mask_padded)
    #     return extract_subsection(temp, shape=mask.shape)

    # @property
    # def inscribed_sphere(self):
    #     dt = self.dt
    #     r = dt.max()
    #     inv_dt = edt(dt < r)
    #     return inv_dt < r

    @property
    def sphericity(self):
        vol = self.volume
        r = (3 / 4 / np.pi * vol)**(1 / 3)
        a_equiv = 4 * np.pi * r**2
        a_region = self.surface_area
        return a_equiv / a_region

    # @property
    # def skeleton(self):
    #     return skeletonize_3d(self.mask)

    @property
    def surface_area(self):
        mask = self.mask
        tmp = np.pad(np.atleast_3d(mask), pad_width=1, mode='constant')
        tmp = spim.convolve(tmp, weights=ball(1)) / 5
        verts, faces, norms, vals = marching_cubes(volume=tmp, level=0)
        verts=verts*self.Vox_size
        self._surface_mesh_vertices = verts
        self._surface_mesh_simplices = faces
        area = mesh_surface_area(verts, faces)
        return area
    
    @property
    def inertia_eigenvecs_eigvals_scaled(self):
        Tensor=self.inertia_tensor
        S = np.diag(self.Vox_size)
        # Calculate the rescaled inertia tensor
        Tensor_rescaled = np.dot(S, np.dot(Tensor, S.T))
        eigenvalues,eigenvectors = np.linalg.eig(Tensor_rescaled)
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices to sort in descending order
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        return np.array(sorted_eigenvectors),np.array(sorted_eigenvalues)

    # @property
    # def surface_mesh_vertices(self):
    #     if not hasattr(self, '_surface_mesh_vertices'):
    #         _ = self.surface_area
    #     return self._surface_mesh_vertices

    # @property
    # def surface_mesh_simplices(self):
    #     if not hasattr(self, '_surface_mesh_simplices'):
    #         _ = self.surface_area
    #     return self._surface_mesh_simplices

    @property
    def convex_volume(self):
        return self.convex_area*self.Vox_size[0]*self.Vox_size[1]*self.Vox_size[2]