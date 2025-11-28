import numpy as np
from matplotlib.path import Path  # Added import for Path

from modules.Tools.brillouin_zone_tools import BZ_tools as bz_tools


class get_BZ_hex:
    def __init__(self, a=None, phi=None):
        if a is None:
            a = 1

        self.a = a
        self.side_bz = 2*np.pi/(3*a) * 2
        length_G_vec = 2*np.pi/(3*a) * np.sqrt(3)

        if isinstance(phi, float) or isinstance(phi, int) or phi is None:
            if phi is  None:
                self.phi = 0 
            else:
                self.phi = phi
        else:
            raise ValueError("phi must be a float")
        
        self.R1 = np.array([np.cos(np.pi/3 - self.phi), - np.sin(np.pi/3 - self.phi)])
        self.R2 = np.array([np.cos(np.pi/3 + self.phi), + np.sin(np.pi/3 + self.phi)])

        self.G1 = length_G_vec * np.array([np.cos(+ np.pi/6 - self.phi), + np.sin(+ np.pi/6 - self.phi)])
        self.G2 = length_G_vec * np.array([np.cos(- np.pi/6 - self.phi), + np.sin(- np.pi/6 - self.phi)])

    def apply_rotation(self, point):
        """Apply rotation by phi to a point
        
        Args:
            point (tuple or np.ndarray): Point (x, y) to rotate
            
        Returns:
            np.ndarray: Rotated point
        """
        x_temp, y_temp = point
        x = np.cos(self.phi) * x_temp - np.sin(self.phi) * y_temp
        y = np.sin(self.phi) * x_temp + np.cos(self.phi) * y_temp
        
        return np.array([x, y])

    def get_BZ_hex(self):
        """Hexagonal Brillouin zone rotated by phi
        Returns:
            dict: Dictionary containing BZ information
        """
        
        self.bz_vertices = [np.array([self.side_bz*np.cos(i*np.pi/3 - self.phi), 
                                      self.side_bz*np.sin(i*np.pi/3 - self.phi)]) for i in range(6)]

        Gamma = np.array([0.0, 0.0])
        K0 = self.bz_vertices[0]
        K1 = self.bz_vertices[1]
        K2 = self.bz_vertices[2]

        HSP = {'Γ'  : Gamma,
               'K'  : K0,
               'K\'': K1,
               '-K' : -K0,
               'M'  : (K0 + K1)/2,
               'M\'': (K1 + K2)/2,
               '-M' : -(K0 + K1)/2}
        
        band_path = {"standard path": ['K', 'Γ', 'M', 'K'],
                    "rotated path": ['K\'', 'Γ',  'M\'', 'K\''],
                    "inverse path": ['-K', 'Γ', '-M', '-K']}
        
        return {"reciprocal_vectors": [self.G1, self.G2],
                "BZ_corners": self.bz_vertices,
                "high_symmetry_points": HSP,
                "band_paths": band_path}
    
    def get_fbz_grid(self, N, center = (0,0), print_idx=True):
        """Get grid points inside the first Brillouin zone
        Args:
            N (int): Number of points along each dimension
            step (float, optional): Step size. Defaults to 0.01.
            buffer (float, optional): Buffer for point containment. Defaults to 0.1.
            print_idx (bool, optional): Whether to return indices. Defaults to True.
        Returns:
            tuple: (grid_points, grid_indices) - Points inside the BZ and their indices
        """
        # Make sure we have the BZ vertices
        if not hasattr(self, 'bz_vertices'):
            self.get_BZ_hex()
            
        x0, y0 = center

        dx = 4 *np.pi/(3 * self.a * N)
        dy = 4 *np.pi/(3 * self.a * N) / (2 * np.sqrt(3))

        length_dk = np.minimum(dx, dy)
        area = dx * dy

        Nx, Ny = N, int(np.ceil(N * (2 * np.sqrt(3)))) + 1

        points = []
        
        if print_idx:
            indices = []

        for i in range(-Ny-1, Ny+1):
            # Offset every other row by half a step
            offset = (i%2) * dx/2
            
            for j in range(-Nx, Nx + 1):
                x_temp = x0 + j * dx + offset
                y_temp = y0 + i * dy 
                
                # Apply rotation using the dedicated function
                point = self.apply_rotation((x_temp, y_temp))
                points.append(point)
                
                if print_idx:
                    indices.append((i,j))

        grid_points = np.array(points)

        if print_idx:
            grid_indices = np.array(indices)
        
        polygon_path = Path(self.bz_vertices)
        inside_mask = polygon_path.contains_points(grid_points, radius = dx * 0.01)

        if print_idx:
            return grid_points[inside_mask], area, grid_indices[inside_mask]
        else: 
            return grid_points[inside_mask], area, None



class get_BZ_tetra:
    def __init__(self, a=None, b=None):
        
        if a is None or b is None:
            a, b = 1, np.sqrt(3)

        self.a = a
        self.b = b
        
        self.R1 = np.array([a, 0])
        self.R2 = np.array([0, b])

        self.G1 = np.array([ 2 * np.pi / a , 0])
        self.G2 = np.array([ 0 , 2 * np.pi / b])
        
        # Add missing Gamma point definition
        self.Gamma = np.array([0.0, 0.0])
        # No phi in Tetra, so set it to 0 to avoid issues with apply_rotation
        self.phi = 0

        print("Reciprocal vectors:")
        print(f"G1: R1 dot G1 = 2 pi and G1 dot R2 = 0 \n{self.G1}")
        print(f"G2: R2 dot G2 = 2 pi and G2 dot R1 = 0 \n{self.G2}")

    def apply_rotation(self, point):
        """Apply rotation by phi to a point
        
        Args:
            point (tuple or np.ndarray): Point (x, y) to rotate
            
        Returns:
            np.ndarray: Rotated point
        """
        x_temp, y_temp = point
        x = np.cos(self.phi) * x_temp - np.sin(self.phi) * y_temp
        y = np.sin(self.phi) * x_temp + np.cos(self.phi) * y_temp
        
        return np.array([x, y])

    def get_BZ_tetra(self):
        """Tetragonal Brillouin zone
        Args:
            a, b (float): lattice constants
        """
        
        BZ = [( + self.G1[0]/2, + self.G2[1]/2),
              ( - self.G1[0]/2, + self.G2[1]/2),
              ( - self.G1[0]/2, - self.G2[1]/2),
              ( + self.G1[0]/2, - self.G2[1]/2)]

        HSP = {'Γ' : self.Gamma,
               'X' :  self.G1/2,
               '-X': -self.G1/2,
               'Y' :  self.G2/2,
               'M' :  (self.G1 + self.G2)/2,
               'M\'': (self.G2 - self.G1)/2,
               '-M': -(self.G1 + self.G2)/2,}
        

        band_path = {"standard path":['X', 'Γ', 'M', 'X'],
                    "rotated path":  ['Y', 'Γ', 'M\'', 'Y'],
                    "inverse path":  ['-X', 'Γ', '-M', '-X']}
        
        return {"reciprocal_vectors": [self.G1, self.G2],
                "BZ_corners": BZ,
                "high_symmetry_points": HSP,
                "band_paths": band_path}
    
    
    def get_fbz_grid(self, N, center = (0,0), print_idx=True):
        """Get grid points inside the first Brillouin zone
        Args:
        N (int): Number of points along each dimension
        step (float, optional): Step size. Defaults to 0.01.
        buffer (float, optional): Buffer for point containment. Defaults to 0.1.
        print_idx (bool, optional): Whether to return indices. Defaults to True.
        Returns:
        tuple: (grid_points, grid_indices) - Points inside the BZ and their indices
        """

        dG1, dG2 = self.G1/(2*N), self.G2/(2*N)


        if center == "shift":
            x0, y0 = - (dG1 + dG2) / 2
        elif center == (0,0):
            x0, y0 = 0, 0
        else:
            x0, y0 = center

        # Fix the area calculation (cross product is for 3D vectors)
        area = np.abs(dG1[0] * dG2[1] - dG1[1] * dG2[0])

        points = []
        
        if print_idx:
            indices = []

        for i in range(- N, N):
            for j in range(- N, N):
                x = i * dG1[0] + j * dG2[0] + x0
                y = i * dG1[1] + j * dG2[1] + y0
                point = np.array([x, y])
                points.append(point)
                    
                if print_idx:
                    indices.append((i,j))

        grid_points = np.array(points)

        if print_idx:
            grid_indices = np.array(indices)

        # Create a polygon path for BZ and filter points
        polygon_vertices = [
            (self.G1[0]/2, self.G2[1]/2),
            (-self.G1[0]/2, self.G2[1]/2),
            (-self.G1[0]/2, -self.G2[1]/2),
            (self.G1[0]/2, -self.G2[1]/2)
        ]
        polygon_path = Path(polygon_vertices)
        inside_mask = polygon_path.contains_points(grid_points)
        
        if print_idx:
            return grid_points[inside_mask], area, grid_indices[inside_mask]
        else: 
            return grid_points[inside_mask], area, None
        
        

class get_BZ_any:
    def __init__(self, r1, r2):
        self.Gamma = np.array([0.0, 0.0])

        if isinstance(r1, (list, tuple)):
            self.R1 = np.array(r1)
        else:
            self.R1 = r1
            
        if isinstance(r2, (list, tuple)):
            self.R2 = np.array(r2)
        else:
            self.R2 = r2
            
        self.G1, self.G2 = bz_tools.reciprocal_vector(self.R1, self.R2)

    def get_BZ_parallelogram(self, G_vector = None):
        """Generate the parallelogram first Brillouin zone (FBZ)."""
        if G_vector is None:
            G1, G2 = self.G1, self.G2
        else: 
            G1, G2 = G_vector

        X = (G1 + G2)/2
        Y = (G1 - G2)/2

        BZ = [X, Y, -X, -Y]

        Gamma = np.array([0.0, 0.0])
        M = + G1/2
        N = - G2/2

        HSP = {'Γ' : Gamma,
               'X' : X, '-X' : - X,
               'Y' : Y, '-Y' : - Y,
               'M' : M, '-M' : - M,
               'N' : N, '-N' : - N}

        band_path = {"standard path":['X', 'Γ', 'M', 'X'],
                    "rotated path":  ['Y', 'Γ', 'N', 'Y'],
                    "inverse path":  ['-X', 'Γ', '-M', '-X']}

        return {"reciprocal_vectors": [G1, G2],
                "BZ_corners": BZ,
                "high_symmetry_points": HSP,
                "band_paths": band_path}

    def get_BZ_wigner_seitz_cell(self, G_vector = None):
        """Generate Wigner-Seitz cell for first Brillouin zone."""
        if G_vector is None:
            G1, G2 = self.G1, self.G2
        else: 
            G1, G2 = G_vector

        fbz_corners = bz_tools.get_wigner_seitz_corners(G1, G2)

        Gamma = np.array([0.0, 0.0])

        K_0 =  np.array(fbz_corners[0])
        M_0 = (np.array(fbz_corners[0]) + np.array(fbz_corners[1]))/2

        K_r =  np.array(fbz_corners[1])
        M_r = (np.array(fbz_corners[1]) + np.array(fbz_corners[2]))/2

        HSP = {
            'Γ'  : Gamma,
            'K'  :   K_0,
            'K\'':   K_r,
            '-K' : - K_0,
            'M'  :   M_0,
            'M\'':   M_r,
            '-M' : - M_0,
        }

        band_path = {"standard path": ['K', 'Γ', 'M', 'K'],
                    "rotated path": ['K\'', 'Γ',  'M\'', 'K\''],
                    "inverse path": ['-K', 'Γ', '-M', '-K']}

        return {
            "reciprocal_vectors": [G1, G2],
            "BZ_corners": fbz_corners,
            "high_symmetry_points": HSP,
            "band_paths": band_path
        }
    
    def get_fbz_grid(self, N, center = (0,0), print_idx=True):
        """Get grid points inside the first Brillouin zone
        Args:
            N (int): Number of points along each dimension
            center (tuple): Center point for the grid
            print_idx (bool, optional): Whether to return indices. Defaults to True.
        Returns:
            tuple: (grid_points, grid_indices) - Points inside the BZ and their indices
        """
        dG1, dG2 = self.G1/(2*N), self.G2/(2*N)

        if center == "shift":
            x0, y0 = - (dG1 + dG2) / 2
        elif center == (0,0):
            x0, y0 = 0, 0
        else:
            x0, y0 = center

        # Fix area calculation
        
        area = np.abs(dG1[0] * dG2[1] - dG1[1] * dG2[0])

        points = []
        
        if print_idx:
            indices = []

        for i in range(- N, N):
            for j in range(- N, N):
                x = i * dG1[0] + j * dG2[0] + x0
                y = i * dG1[1] + j * dG2[1] + y0
                point = np.array([x, y])
                points.append(point)
                    
                if print_idx:
                    indices.append((i,j))

        grid_points = np.array(points)

        if print_idx:
            grid_indices = np.array(indices)

        if print_idx:
            return grid_points, area, grid_indices
        else: 
            return grid_points, area, None