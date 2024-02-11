#this one iterates P2. It's quite messy and full of random comments, but does exactly what it says on the tin

import oommfc as oc
import discretisedfield as df
import micromagneticmodel as mm
import micromagneticdata as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.colors as colors
import os

# Conversion factors
Oe = 1000/(4*np.pi)     # conversion Oe->A/m 79.577471/ 1 mT->10 Oe

# Initial magnetization
m0 = (1, 0, 0)   # Initial reduced magnetization
Hx=550 * Oe
#Hy_list=[(n+11)*50*Oe for n in range(10)]
Hz=0
Hy_list = [5500]
Hy_list = [Hy * Oe for Hy in Hy_list]


# magnetic parametes
alpha_YIG = 1.75e-4    # Gilbert damping (-)
Ms_YIG = 140.7e3           # Saturation magnetisation (A/m). 
A_YIG = 4.22e-12           # Exchange stiffness (J/m)
l_ex = (2*A_YIG/(mm.consts.mu0*Ms_YIG**2)) # Exchange length (m)
a_YIG=1.2373e-9  #YIG lattice parameter
BZ=2*np.pi/a_YIG
# Geometry [m]
l = 60e-6
w = 500e-9
t = 50e-9
# self absorbing boundary condition
band = 1e-6
#source
sourceWidth=600e-9 #cercare bene la misura sul righello
sourcePos=-l/4 #almeno iniettiamo le waves nel bulk, possiamo modificare
# Mesh cell [m]
cx = 10e-9   #50e-9
cy = w  #50e-9
cz = t
cell = (cx, cy, cz)


# Amplitude of RF exiting field
H_RF_val=10 * Oe

mn = oc.MinDriver()           # minimization driver
td = oc.TimeDriver()          # time driver

T = 100e-9 #100e-9
f_MAX = 15e9

f_Nyquist = 2*f_MAX
n_Nyquist = T*f_Nyquist
n_oversampling = 50
sampling = int(n_Nyquist+n_oversampling)
t_0 = T/sampling

def alpha_abs(point):
    x, y, z = point
    if (-l/2 < x < -l/2+band):
        return (((x+l/2-band)**2)/(band)**2+alpha_YIG)

    if (l/2-band < x < l/2):
        return (((x-l/2+band)**2)/(band)**2+alpha_YIG)

    else:
        return (alpha_YIG)

# macro to return correct saturation magnetisation inside or outside the sample
def Ms_value(pos):
    # from 0, only inside the sample
    x, y, z = pos

    if (-l/2 < x < l/2 and -w/2 < y < w/2 and 0 < z < t):                                        # rect
        return Ms_YIG

    else:                                                                    # empty space
        return 0


def defSys():
    system = mm.System(name=sysName)
    region = df.Region(p1=(-l/2, -w/2, 0), p2=(l/2, w/2, t))
    mesh = df.Mesh(region=region, cell=cell)
    alpha = df.Field(mesh, nvdim=1, value=alpha_abs)
    system.m = df.Field(mesh, nvdim=3, value=m0, norm=Ms_value)
    damping = mm.Damping(alpha=alpha_YIG)
    system.dynamics = mm.Precession(gamma0=mm.consts.gamma0) + damping
    return system, region, mesh, alpha

def sysEnergy(Hx,Hy,Hz,system):
    
    # Zeeman field
    H_DC = (Hx, Hy, Hz)
    # demagnetizing energy
    dem = mm.Demag()
    # exchange energy
    ex = mm.Exchange(A=A_YIG)
    # zeeman energy
    zem = mm.Zeeman(H=H_DC, name='bias')
    system.energy = dem + ex + zem


def checkSys(system):


    fig,ax =plt.subplots(figsize=(25,10))

    plt.title('$\hat m_x$')
    """
    system.m accesses the megnetization
    m.x accesses the x component
    x.sel("z") means "sliced at a fixed value of z", by defaut the middle of the system
    from mpl.scalar on it's just about putting the red/blue color code on the right
    """
    system.m.x.sel('z').mpl.scalar(ax=ax,vmin=-Ms_YIG,vmax=Ms_YIG,cmap='seismic')

    system.m.sel('z').resample((20,10)).mpl.vector(ax=ax,headwidth=3,scale=2e7)
    """
    this basically just shows the vaues of the vector field m, as colored dots/arrows. 
    Uses the matplotlib.quiver method
    for low fields you only see yellow dots since the equilibrium magnetization is parallel to z
    """
    plt.savefig(f'{sysName}/images/mx.png', bbox_inches='tight')


    fig,ax =plt.subplots(figsize=(25,10))
    plt.title('$\hat m_y$')
    system.m.y.sel('z').mpl.scalar(ax=ax,vmin=-Ms_YIG,vmax=Ms_YIG,cmap='seismic')
    system.m.sel('z').resample((20,10)).mpl.vector(ax=ax,headwidth=3,scale=2e7)
    plt.savefig(f'{sysName}/images/my.png', bbox_inches='tight')

    fig,ax =plt.subplots(figsize=(25,10))
    plt.title('$\hat m_z$')
    system.m.z.sel('z').mpl.scalar(ax=ax,vmin=-1000,vmax=1000,cmap='seismic')
    system.m.sel('z').resample((20,10)).mpl.vector(ax=ax,headwidth=3,scale=2e7)
    plt.savefig(f'{sysName}/images/mz.png', bbox_inches='tight')

def getEq(system):
    # get system to equilibrium
    mn.drive(system)

def Hspace_RF(point):
    x, y, z = point
    if (sourcePos-sourceWidth/2< x < sourcePos+sourceWidth/2 and -w/2 < y < w/2 and 0 < z < t):
        return (0, 0, H_RF_val)
    else:
        return (0,0,0)


def injectRF(mesh, system):
    H_RF = df.Field(mesh, nvdim=3, value=Hspace_RF)
    zemRF = mm.Zeeman(H=H_RF, func='sinc', f=f_MAX, t0=T/sampling, name='RF')
    try:
        system.energy += zemRF
    finally:
        td.drive(system, t=T, n=sampling, n_threads=19, verbose=2)

def dataProcessing():
    data = md.Data(sysName)[-1]
    array = data.to_xarray()
    data_np = np.array(array)

    mx = data_np[:, round((band)/cx):round((l-band)/cx), 0, 0, 0]
    my = data_np[:, round((band)/cx):round((l-band)/cx), 0, 0, 1]
    mz = data_np[:, round((band)/cx):round((l-band)/cx), 0, 0, 2]
    # questo non ho ben capito a cosa serva, nel senso che Ax è letteralmente uguale a mx, almeno nei miei test
    Ax = np.reshape(mx, (-1, round(l/cx-2*band/cx)))
    Ax = np.fliplr(Ax)
    # 2d perchè ci serve lungo il primo asse (tempo) e lungo il secondo (spazio)
    m_fft_x = np.fft.fft2(Ax)
    m_fft_x = np.fft.fftshift(m_fft_x)  # questo centra le frequenze

    Ay = np.reshape(my, (-1, round(l/cx-2*band/cx)))
    Ay = np.fliplr(Ay)
    m_fft_y = np.fft.fft2(Ay)
    m_fft_y = np.fft.fftshift(m_fft_y)

    Az = np.reshape(mz, (-1, round(l/cx-2*band/cx)))
    Az = np.fliplr(Az)
    m_fft_z = np.fft.fft2(Az)
    m_fft_z = np.fft.fftshift(m_fft_z)
    return m_fft_x, m_fft_y, m_fft_z

def getDispersions(m_fft_x,m_fft_y,m_fft_z):

    # Show the intensity plot of the 2D FFT

    #mx
    plt.figure(figsize=(10, 10))
    plt.title('F(mx)')
    extent = [-1/cx, 1/cx, -f_MAX, f_MAX]  # extent of k values and frequencies
    plt.imshow(np.log(np.abs(m_fft_x)**2), extent=extent,
            aspect='auto', origin='lower', norm=colors.CenteredNorm(vcenter=22), cmap="inferno")
    plt.ylabel("$f$ (Hz)")
    plt.xlabel("$kx$ (1/m)")
    plt.xlim([-10e6, 10e6])
    plt.ylim([3e9, f_MAX])
    plt.savefig(f'{sysName}/images/F(mx).png', bbox_inches='tight')

    #my
    plt.figure(figsize=(10, 10))
    plt.title('F(my)')
    extent = [-1/cx, 1/cx, -f_MAX, f_MAX]  # extent of k values and frequencies
    plt.imshow(np.log(np.abs(m_fft_y)**2), extent=extent,
            aspect='auto', origin='lower', cmap="inferno")
    plt.ylabel("$f$ (Hz)")
    plt.xlabel("$kx$ (1/m)")
    plt.xlim([-10e6, 10e6])
    plt.ylim([3e9, f_MAX])
    plt.savefig(f'{sysName}/images/F(my).png', bbox_inches='tight')

    #mz
    plt.figure(figsize=(10, 10))
    plt.title('F(mz)')
    extent = [-1/cx, 1/cx, -f_MAX, f_MAX]  # extent of k values and frequencies
    plt.imshow(np.log(np.abs(m_fft_z)**2), extent=extent,
            aspect='auto', origin='lower', cmap="inferno")
    plt.xlim([-10e6, 10e6])
    plt.ylim([3e9, f_MAX])
    plt.ylabel("$f$ (Hz)")
    plt.xlabel("$kx$ (1/m)")

    plt.savefig(f'{sysName}/images/F(mz).png', bbox_inches='tight')

def saveParams(sysName, Hy):
    with open(f"{sysName}/params.txt",'w') as f:
        if cy==w and cz==t:
            f.write(f"cell=({cx},w,t)\n")
        elif cy==w:
            f.write(f"cell=({cx},w,{cz})\n")
        elif cz==t:
            f.write(f"cell=({cx},{cy},t)\n")
        else:
            f.write(f"cell=({cx},{cy},{cz})")
        f.write("\n")
        f.write("Bias field:\n")
        f.write(f"Hx={Hx/Oe}Oe\n")
        f.write(f"Hy={Hy/Oe}Oe\n")
        f.write(f"Hz={Hz/Oe}Oe\n")
        f.write("\n")
        f.write("Exitation field:\n")
        f.write(f"Amplitude={H_RF_val/Oe}Oe\n")
        f.write(f"f_MAX={f_MAX}\n")

for Hy in Hy_list:
    sysName=f"P2_{int(T*1e9)}ns_{int(f_MAX*1e-9)}GHz_{int(Hy/Oe)}Oe"
    system, region, mesh, alpha = defSys()
    sysEnergy(Hx,Hy,Hz,system)
    getEq(system)
    directory="images"
    parentDir=f"{sysName}"
    path=os.path.join(parentDir,directory)

    if not os.path.isdir(path):
        os.mkdir(path)
    checkSys(system)
    injectRF(mesh,system)
    m_fft_x, m_fft_y, m_fft_z=dataProcessing()
    getDispersions(m_fft_x, m_fft_y, m_fft_z)
    saveParams(sysName, Hy)
    plt.close('all')