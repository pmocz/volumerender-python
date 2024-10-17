from jax import jit
import jax.numpy as jnp

import time 
import matplotlib.pyplot as plt
import h5py as h5

from interpax import interp3d

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""

@jit
def transferFunction(x):
    r = 1.0*jnp.exp( -(x - 9.0)**2/1.0 ) +  0.1*jnp.exp( -(x - 3.0)**2/0.1 ) +  0.1*jnp.exp( -(x - -3.0)**2/0.5 )
    g = 1.0*jnp.exp( -(x - 9.0)**2/1.0 ) +  1.0*jnp.exp( -(x - 3.0)**2/0.1 ) +  0.1*jnp.exp( -(x - -3.0)**2/0.5 )
    b = 0.1*jnp.exp( -(x - 9.0)**2/1.0 ) +  0.1*jnp.exp( -(x - 3.0)**2/0.1 ) +  1.0*jnp.exp( -(x - -3.0)**2/0.5 )
    a = 0.6*jnp.exp( -(x - 9.0)**2/1.0 ) +  0.1*jnp.exp( -(x - 3.0)**2/0.1 ) + 0.01*jnp.exp( -(x - -3.0)**2/0.5 )
    
    return r, g, b, a

@jit
def get_query_points(angle, qx, qy, qz):
    qxR = qx
    qyR = qy * jnp.cos(angle) - qz * jnp.sin(angle) 
    qzR = qy * jnp.sin(angle) + qz * jnp.cos(angle)

    return qxR, qyR, qzR 

@jit
def update_color(a, x, image_x):
    color = a * x + (1 - a) * image_x

    return color 

def main():
    """ Volume Rendering """
    
    # Load Datacube
    f = h5.File('C12_Beta2_256_0060.h5', 'r')
    datacube = jnp.array(f['density'])    
    print('jnp.shape(datacube)',jnp.shape(datacube))

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = jnp.linspace(-Nx/2, Nx/2, Nx)
    y = jnp.linspace(-Ny/2, Ny/2, Ny)
    z = jnp.linspace(-Nz/2, Nz/2, Nz)
    
    # Do Volume Rendering at Different Viewing Angles
    N = 180
    c = jnp.linspace(-N/2, N/2, N)
    qx, qy, qz = jnp.meshgrid(c,c,c)

    t0 = time.time()

    Nangles = 90

    for i in range(Nangles):        
        print(f"\rRendering Scene {i + 1} of {Nangles}", end="", flush=True)
        angle = jnp.pi / 2 * i / Nangles
    
        # image = render_scene(angle, qx, qy, qz, x, y, z, datacube, N)
    
        # Camera Grid / Query Points -- rotate camera view
        qxR, qyR, qzR = get_query_points(angle, qx, qy, qz)

        # Interpolate onto Camera Grid
        camera_grid = interp3d(xq=qxR.ravel(), yq=qyR.ravel(), zq=qzR.ravel(), 
                                x=x, y=y, z=z, 
                                f=datacube, method='linear').reshape((N,N,N))
        
        # Do Volume Rendering
        image_r = jnp.zeros((camera_grid.shape[1],camera_grid.shape[2]))
        image_g = jnp.zeros((camera_grid.shape[1],camera_grid.shape[2]))
        image_b = jnp.zeros((camera_grid.shape[1],camera_grid.shape[2]))

        for dataslice in camera_grid:
            r,g,b,a = transferFunction(jnp.log(dataslice))
            image_r = update_color(a, r, image_r)
            image_g = update_color(a, g, image_g)
            image_b = update_color(a, b, image_b)
        
        image_r = jnp.clip(image_r,0.0,1.0)
        image_g = jnp.clip(image_g,0.0,1.0)
        image_b = jnp.clip(image_b,0.0,1.0)
        
        image = jnp.dstack([image_r, image_g, image_b])
        image = jnp.clip(image,0.0,1.0)        

        # Plot Volume Rendering
        plt.figure(figsize=(4,4), dpi=80)
        # from astropy.visualization.mpl_normalize import simple_norm
        # norm = simple_norm(image, stretch='asinh', asinh_a=0.01)
        # plt.imshow(image, norm=norm)
        plt.imshow(image)
        plt.axis('off')        
        plt.savefig(f"render/volumerender_jax{i:03d}.jpg", dpi=240, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"time {(time.time() - t0)} s")
    print(f'frame_times: mean {jnp.mean(frame_times)}\nall: {frame_times}')

    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=80)
    plt.imshow(jnp.log(jnp.mean(datacube,0)), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    plt.savefig('render/projection_jax.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    plt.show()
    
    # import subprocess
    # subprocess.call(f'''ffmpeg -i render/volumerender_jax%03d.jpg -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an video.mp4'''.split())
    # ffmpeg -i render/volumerender_jax%03d.jpg -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an video.mp4

    return 0

  
if __name__== "__main__":
  main()
