import numpy as np
import pyvista as pv


from path import *

class sub_manifold:
    def __init__(self, manifold:pv.PolyData, name):
        self.manifold = manifold
        self.name = name

    def Initial_guess(self,target_MF:pv.PolyData):
        pass

    def project(self,target_MF:pv.PolyData):
        pass

    def getTangentDeriv(self, derivatives):
        pass

    def getTransferedSMf(self,target_MF:pv.PolyData):
        pass

class sub_manifold_0(sub_manifold):
    def __init__(self, manifold, name, point):
        super().__init__(manifold, name)
        self.point = point

    def project(self,target_MF:sub_manifold):
        self.manifold.point_data['deformed'][self.point]=target_MF.manifold.points[target_MF.point]

    def Initial_guess(self,target_MF:sub_manifold):
        return self.project(target_MF)
    
    def getTangentDeriv(self, derivatives):
        derivatives[self.point]=0
        return derivatives
    
    def getTransferedSMf(self,target_MF:pv.PolyData):
        p=target_MF.find_closest_point(self.point)
        return sub_manifold_0(target_MF,self.name,p)

class sub_manifold_1_closed(sub_manifold):
    def __init__(self, manifold, name, path):
        super().__init__(manifold, name)
        self.path = path

    def project(self,target_MF:sub_manifold):
        #print('project')
        target_points = target_MF.manifold.points[target_MF.path]
        target_ring=np.vstack([target_points, target_points[0]])

        init_points = self.manifold.point_data['deformed'][self.path]
        
        newPoints=np.zeros((init_points.shape[0],3))
        for i in range(init_points.shape[0]):
            #print(i)
            point_init = init_points[i,:]
            point_on_line,_=closest_point_on_path(target_ring,point_init)
            newPoints[i,:]=point_on_line
        self.manifold.point_data['deformed'][self.path]=newPoints
        
    def Initial_guess(self,target_MF:sub_manifold):
        target_points = target_MF.manifold.points[target_MF.path]   
        target_ring=np.vstack([target_points, target_points[0]])
        target_ring_length=dist_from_start(target_ring)[-1]

        init_points = self.manifold.point_data['rotated'][self.path]   
        init_ring=np.vstack([init_points, init_points[0]])
        init_ring_length=dist_from_start(init_ring)[-1]

        point_init = self.manifold.point_data['rotated'][self.path[0]]
        offset=closest_position_on_path(target_ring,point_init)


        #point_on_line=target_ring.interpolate(target_ring.project(point_init))
        
        interpolated_points=interpolate_on_path(target_ring,(dist_from_start(init_points)*target_ring_length/init_ring_length+offset)%target_ring_length)
    
        point_data = np.full((self.manifold.n_points,3), np.nan)
        point_data[self.path] = interpolated_points
        self.manifold.point_data['deformed']=point_data
        
        return 

    def getTangentDeriv(self, derivatives):
        ring_points=self.manifold.point_data['deformed'][self.path]
        tangent_vectors = np.roll(ring_points, -1, axis=0) - ring_points
        tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]

        tangent_components = np.sum(derivatives[self.path] * tangent_vectors, axis=1)[:,np.newaxis] * tangent_vectors
        derivatives[self.path]=tangent_components

        return derivatives

    def getTransferedSMf(self,target_MF:pv.PolyData):
        target_path=[]
        for i in range(self.path.shape[0]):
            p=target_MF.find_closest_point(self.path[i])
            if p == target_path[-1]:
                continue
            target_path.append(p)
        return sub_manifold_1_closed(target_MF,self.name,np.array(target_path,dtype=int)) 

class sub_manifold_1_open(sub_manifold):
    def __init__(self, manifold, name, path):
        super().__init__(manifold, name)
        self.path = path

    def project(self,target_MF:sub_manifold):
        target_string = target_MF.manifold.points[target_MF.path]


        init_points = self.manifold.point_data['rotated'][self.path]
        
        newPoints=np.zeros((init_points.shape[0],3))
        for i in range(init_points.shape[0]):
            #print(i)
            point_init = init_points[i,:]
            rel_pos=closest_position_on_path(target_string,point_init)
            if rel_pos<1e-5:
                newPoints[i,:]=init_points[i,:]
            elif rel_pos>1-1e-5:
                newPoints[i,:]=init_points[i,:]
            else:
                point_on_line=target_string. interpolate (rel_pos,normalized=True)
                newPoints[i,:]=np.array((point_on_line.x,point_on_line.y,point_on_line.z))
        self.manifold.point_data['deformed'][self.path]=newPoints
        pass

    def Initial_guess(self,target_MF:sub_manifold):
        super().describe()
        print(f"Dimension: 1 (Open)")
        print(f"Path: {self.path}")

        return self.project(target_MF)
    
    def getTransferedSMf(self,target_MF:pv.PolyData):
        target_path=[]
        for i in range(self.path.shape[0]):
            p=target_MF.find_closest_point(self.path[i])
            if p == target_path[-1]:
                continue
            target_path.append(p)
        return sub_manifold_1_open(target_MF,self.name,np.array(target_path,dtype=int)) 