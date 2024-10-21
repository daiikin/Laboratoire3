#%%

class Cylinder :
    def __init__(self, length, height, radius):
        self.length = length 
        self.height = height
        self.radius = radius
#%%
print("hihihi")
hi = 3
#%%


class sphere:
    def __init__(self, radius):
        self.radius = radius

    def volume(self):
        return 4/3*3.14*self.radius**3
    
    def surface_area(self):
        return 4*3.14*self.radius**2
    
    def __str__(self):

        return f"Sphere with radius {self.radius}"
    
    def __add__(self, other):

        return sphere(self.radius + other.radius)
    
    def __lt__(self, other):

        return self.radius < other.radius

#%%


#%%
