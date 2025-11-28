import numpy as np

class BZ_tools:
    @staticmethod
    def reciprocal_vector(a1, a2):
        a1x, a1y = a1
        a2x, a2y = a2

        a1 = np.array([a1x, a1y, 0.0])
        a2 = np.array([a2x, a2y, 0.0])
        a3 = np.array([0.0, 0.0, 1.0])

        G1 = np.cross(a2, a3)
        G2 = np.cross(a3, a1)

        vol = np.dot(np.cross(a1, a2), a3)
        G1 = G1 * (2 * np.pi / vol)
        G2 = G2 * (2 * np.pi / vol)
        
        return G1[:2], G2[:2]


    @staticmethod
    def sorting_vectors_counter_clockwise(vectors, angle_origin = 0):
        """Sort vectors in counter-clockwise order"""
        angles = []

        for vector in vectors:
            vx, vy = vector
            angle = (np.arctan2(vy, vx) + 2*np.pi - angle_origin) % (2*np.pi)  # ensure 0 < angle < 2 pi
            angles.append(angle)

        idx = np.argsort(angles)
        vectors = np.array(vectors)[idx]
        return vectors
      
    @classmethod
    def get_nearest_lattices(cls, v1, v2):
        """Find nearest lattices"""

        dot_product = np.dot(v1, v2)
        nearest_lattices = [v1, v2, - v1, - v2]

        if np.isclose(dot_product, 0):      # check floating number
            return cls.sorting_vectors_counter_clockwise(nearest_lattices)

        else:
            if   np.sign(dot_product) == 1:
                nearest_lattices.append(v1 - v2)
                nearest_lattices.append(v2 - v1)
            
            elif np.sign(dot_product) == -1:
                nearest_lattices.append(  v1 + v2)
                nearest_lattices.append(-(v1 + v2))
        
            return cls.sorting_vectors_counter_clockwise(nearest_lattices)


    @staticmethod
    def find_intersetion_of_perp_plane(v1: np.ndarray, v2: np.ndarray):
        """Find intersection point solving: dot(G_i, k) = G_i^2/2"""
        g1_sq = np.sum(v1**2)
        g2_sq = np.sum(v2**2)
        gmat = np.array([v1, v2])
        return np.array([g1_sq/2, g2_sq/2]) @ np.linalg.inv(gmat).T


    @classmethod
    def get_wigner_seitz_corners(cls, G1, G2):
        
        """Calculate corners of Brillouin zone"""
        
        sorted_bz_corners = cls.get_nearest_lattices(G1, G2)
        corners = []

        for j in range(len(sorted_bz_corners)):
            g1 = sorted_bz_corners[j-1]
            g2 = sorted_bz_corners[j]

            corner = tuple(cls.find_intersetion_of_perp_plane(g1, g2))
            corners.append(corner)
            
        corners = cls.sorting_vectors_counter_clockwise(corners)
        return corners
    
    @classmethod
    def get_minimal_lattices(cls, r1, r2):
        return