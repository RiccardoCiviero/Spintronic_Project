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
Hy=0
Hy_list = [0, 50, 250, 550, 1100, 2200]
Hy_list = [Hy * Oe for Hy in Hy_list]
Hz=0

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
f_MAX = 4e9
f_MAX_list=[3.2e9, 3.5e9, 3.8e9, 4e9]
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

def sysEnergy(Hx,Hy,Hz,Hy_transverse,system):
    
    # Zeeman field
    H_DC = (Hx, Hy, Hz)
    # demagnetizing energy
    dem = mm.Demag()
    # exchange energy
    ex = mm.Exchange(A=A_YIG)
    # zeeman energy
    zem = mm.Zeeman(H=H_DC, name='bias')
    zemTransverse=mm.Zeeman(H=Hy_transverse, name='transverse')
    system.energy = dem + ex + zem + zemTransverse



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


def injectRF(mesh, system, freq=f_MAX):
    H_RF = df.Field(mesh, nvdim=3, value=Hspace_RF)
    zemRF = mm.Zeeman(H=H_RF, func='sin', f=freq, t0=T/sampling, name='RF')
    try:
        system.energy += zemRF
    finally:
        td.drive(system, t=T, n=sampling, n_threads=19, verbose=2)

def dataProcessing(side="all"):
    data = md.Data(sysName)[-1]
    array = data.to_xarray()
    data_np = np.array(array)
    transverseFieldSize=0.5e-6
    buffer=transverseFieldSize/10
    if side=="all":
        mx = data_np[:, round((band)/cx):round((l-band)/cx), 0, 0, 0]
        my = data_np[:, round((band)/cx):round((l-band)/cx), 0, 0, 1]
        mz = data_np[:, round((band)/cx):round((l-band)/cx), 0, 0, 2]
        halfsize=round(l/cx-2*band/cx)
    elif side=="left":
        mx = data_np[:, round((band)/cx):round((l/2-transverseFieldSize/2-buffer/2)/cx), 0, 0, 0]
        my = data_np[:, round((band)/cx):round((l/2-transverseFieldSize/2-buffer/2)/cx), 0, 0, 1]
        mz = data_np[:, round((band)/cx):round((l/2-transverseFieldSize/2-buffer/2)/cx), 0, 0, 2]
        halfsize=round((l/2-transverseFieldSize/2-buffer/2)/cx)-round((band)/cx)
    elif side=="right":
        mx = data_np[:, round((l/2+transverseFieldSize/2+buffer/2)/cx):round((l-band)/cx), 0, 0, 0]
        my = data_np[:, round((l/2+transverseFieldSize/2+buffer/2)/cx):round((l-band)/cx), 0, 0, 1]
        mz = data_np[:, round((l/2+transverseFieldSize/2+buffer/2)/cx):round((l-band)/cx), 0, 0, 2]
        halfsize=round((l/2-transverseFieldSize/2-buffer/2)/cx)-round((band)/cx)
    elif side=="center":
        mx = data_np[:, round((l/2-transverseFieldSize/2)/cx):round((l/2+transverseFieldSize/2)/cx), 0, 0, 0]
        my = data_np[:, round((l/2-transverseFieldSize/2)/cx):round((l/2+transverseFieldSize/2)/cx), 0, 0, 1]
        mz = data_np[:, round((l/2-transverseFieldSize/2)/cx):round((l/2+transverseFieldSize/2)/cx), 0, 0, 2]
        halfsize=round((transverseFieldSize/2)/cx)
    # questo non ho ben capito a cosa serva, nel senso che Ax è letteralmente uguale a mx, almeno nei miei test
    Ax = np.reshape(mx, (-1, halfsize))
    Ax = np.fliplr(Ax)
    # 2d perchè ci serve lungo il primo asse (tempo) e lungo il secondo (spazio)
    m_fft_x = np.fft.fft2(Ax)
    m_fft_x = np.fft.fftshift(m_fft_x)  # questo centra le frequenze

    Ay = np.reshape(my, (-1, halfsize))
    Ay = np.fliplr(Ay)
    m_fft_y = np.fft.fft2(Ay)
    m_fft_y = np.fft.fftshift(m_fft_y)

    Az = np.reshape(mz, (-1, halfsize))
    Az = np.fliplr(Az)
    m_fft_z = np.fft.fft2(Az)
    m_fft_z = np.fft.fftshift(m_fft_z)
    return m_fft_x, m_fft_y, m_fft_z

def getDispersions(m_fft_x,m_fft_y,m_fft_z,side="full"):
    xbounds=(-1/(5*cx),1/(5*cx))
    ybounds=(0,f_MAX)
    # Show the intensity plot of the 2D FFT

    #mx
    plt.figure(figsize=(10, 10))
    plt.title('F(mx)')
    extent = [-1/cx, 1/cx, -f_MAX, f_MAX]  # extent of k values and frequencies
    plt.imshow(np.log(np.abs(m_fft_x)**2), extent=extent,
            aspect='auto', origin='lower', norm=colors.CenteredNorm(vcenter=22), cmap="inferno")
    plt.ylabel("$f$ (Hz)")
    plt.xlabel("$kx$ (1/m)")
    plt.xlim(xbounds)
    plt.ylim(ybounds)
    plt.savefig(f'{sysName}/images/F(mx)_{side}.png', bbox_inches='tight')

    #my
    plt.figure(figsize=(10, 10))
    plt.title('F(my)')
    extent = [-1/cx, 1/cx, -f_MAX, f_MAX]  # extent of k values and frequencies
    plt.imshow(np.log(np.abs(m_fft_y)**2), extent=extent,
            aspect='auto', origin='lower', cmap="inferno")
    plt.ylabel("$f$ (Hz)")
    plt.xlabel("$kx$ (1/m)")
    plt.xlim(xbounds)
    plt.ylim(ybounds)
    plt.savefig(f'{sysName}/images/F(my)_{side}.png', bbox_inches='tight')

    #mz
    plt.figure(figsize=(10, 10))
    plt.title('F(mz)')
    extent = [-1/cx, 1/cx, -f_MAX, f_MAX]  # extent of k values and frequencies
    plt.imshow(np.log(np.abs(m_fft_z)**2), extent=extent,
            aspect='auto', origin='lower', cmap="inferno")
    plt.ylabel("$f$ (Hz)")
    plt.xlabel("$kx$ (1/m)")
    plt.xlim(xbounds)
    plt.ylim(ybounds)
    plt.savefig(f'{sysName}/images/F(mz)_{side}.png', bbox_inches='tight')

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

def getTimeEvol(side):
    time=system.table.data["t"].values #.values returns a numpy array-> FFT
    data=md.Data(sysName) #this contains all the drives up to now. [-1] means the last drive. Check the folder "Py_disk_FMR"
    array=data[-1].to_xarray()
    if(side=="left"):
        mz1=array[:, int((3/8*l)/cx), int(w/2/cy), 0, 2]
        mx1=array[:, int((3/8*l)/cx), int(w/2/cy), 0, 0]
        my1=array[:, int((3/8*l)/cx), int(w/2/cy), 0, 1]
    elif side=="center":
        mz1=array[:, int((l/2)/cx), int(w/2/cy), 0, 2]
        mx1=array[:, int((l/2)/cx), int(w/2/cy), 0, 0]
        my1=array[:, int((l/2)/cx), int(w/2/cy), 0, 1]
    elif side=="right":
        mz1=array[:, int((5/8*l)/cx), int(w/2/cy), 0, 2]
        mx1=array[:, int((5/8*l)/cx), int(w/2/cy), 0, 0]
        my1=array[:, int((5/8*l)/cx), int(w/2/cy), 0, 1]
    #mx(t)
    dmx=[i-mx1[0] for i in mx1]
    plt.figure()
    plt.title('$\hat m_x(t)$')
    plt.plot(time,dmx)
    plt.xlabel('t (ns)')
    plt.ylabel('mx average')
    plt.legend(['mx'])
    plt.savefig(f"{sysName}/images/mx(t)_{side}")
    amp=np.abs(np.fft.fft(dmx))**2
    f_axis=np.fft.fftfreq(sampling,d=T/sampling)
    plt.plot(f_axis[0:round(sampling/2)]/1e9, amp[0:round(sampling/2)])
    plt.savefig(f"{sysName}/images/mx(t)_{side}_fft")

    #my(t)
    dmy=[i-my1[0] for i in my1]
    plt.figure()
    plt.title('$\hat m_y(t)$')
    plt.plot(time,dmy)
    plt.xlabel('t (ns)')
    plt.ylabel('my average')
    plt.legend(['my'])
    plt.savefig(f"{sysName}/images/my(t)_{side}")
    amp=np.abs(np.fft.fft(dmy))**2
    f_axis=np.fft.fftfreq(sampling,d=T/sampling)
    plt.plot(f_axis[0:round(sampling/2)]/1e9, amp[0:round(sampling/2)])
    plt.savefig(f'{sysName}/images/my(t)_{side}_fft')



    #mz
    dmz=[i-mz1[0] for i in mz1]
    plt.figure()
    plt.title('$\hat m_z(t)$')
    plt.plot(time,dmz)
    plt.xlabel('t (ns)')
    plt.ylabel('mz average')
    plt.legend(['mz'])
    plt.savefig(f"{sysName}/images/mz(t)_{side}")
    amp=np.abs(np.fft.fft(dmz))**2
    f_axis=np.fft.fftfreq(sampling,d=T/sampling)
    plt.plot(f_axis[0:round(sampling/2)]/1e9, amp[0:round(sampling/2)])
    plt.savefig(f"{sysName}/images/mz(t)_{side}_fft")

    return mx1[0], my1[0], mz1[0]

def getTramissions(baseline):
    data=md.Data(sysName) #this contains all the drives up to now. [-1] means the last drive. Check the folder "Py_disk_FMR"
    array=data[-1].to_xarray()
    mz1_left=array[:, int((3/8*l)/cx), int(w/2/cy), 0, 2]
    my1_left=array[:, int((3/8*l)/cx), int(w/2/cy), 0, 1]
    mz1_right=array[:, int((5/8*l)/cx), int(w/2/cy), 0, 2]
    my1_right=array[:, int((5/8*l)/cx), int(w/2/cy), 0, 1]
    T1z=np.max(mz1_right)**2/np.max(mz1_left)**2
    T1y=np.max(my1_right)**2/np.max(my1_left)**2
    my_BL, mz_BL=baseline
    T2z=np.max(mz1_right)**2/mz_BL**2
    T2y=np.max(my1_right)**2/my_BL**2
    return T1z, T1y, T2z, T2y   

def getSpaceEvol(mx0,my0,mz0):
    numCells=int((l-2*band)/cx)
    space= np.linspace(-l/2+band,l/2-band,numCells)
    print(np.shape(space))
    data=md.Data(sysName) #this contains all the drives up to now. [-1] means the last drive. Check the folder "Py_disk_FMR"
    array=data[-1].to_xarray()
    #print(len(list(data)))


    value_z=array[sampling-1, round((band)/cx):round((l-band)/cx), int(w/2/cy), 0,2]
    value_x=array[sampling-1, round((band)/cx):round((l-band)/cx), int(w/2/cy), 0,0]
    value_y=array[sampling-1, round((band)/cx):round((l-band)/cx), int(w/2/cy), 0,1]
    print(np.shape(value_x))
    dmx=[i-mx0 for i in value_x]
    dmy=[i-my0 for i in value_y]
    dmz=[i-mz0 for i in value_z]

    #mx(x)
    plt.figure()
    plt.title('$\hat m_x(x)$')
    plt.plot(space[1:400],dmx[1:400])
    plt.xlabel('x')
    plt.ylabel('mx average')
    plt.legend(['mx'])
    plt.savefig(f"{sysName}/images/mx(x)")
    amp=np.abs(np.fft.fft(dmx))**2
    f_axis=np.fft.fftfreq(numCells,d=cx)
    plt.plot(f_axis/1e9, amp)
    plt.savefig(f"{sysName}/images/mx(x)_fft")

    #my(x)
    plt.figure()
    plt.title('$\hat m_y(x)$')
    plt.plot(space,dmy)
    plt.xlabel('x')
    plt.ylabel('my average')
    plt.legend(['my'])
    plt.savefig(f"{sysName}/images/my(x)")
    amp=np.abs(np.fft.fft(dmy))**2
    f_axis=np.fft.fftfreq(numCells,d=cx)
    plt.plot(f_axis/1e9, amp)
    plt.savefig(f"{sysName}/images/mx(y)_fft")

    #mz(x)
    plt.figure()
    plt.title('$\hat m_z(x)$')
    plt.plot(space,dmz)
    plt.xlabel('x ')
    plt.ylabel('mz average')
    plt.legend(['mz'])
    plt.savefig(f"{sysName}/images/mz(x)")
    amp=np.abs(np.fft.fft(dmz))**2
    f_axis=np.fft.fftfreq(numCells,d=cx)
    plt.plot(f_axis/1e9, amp)
    plt.savefig(f"{sysName}/images/mz(x)_fft")

Ty1=[]
Ty2=[]
Tz1=[]
Tz2=[]

with open(f"./Transmissions.csv",'a') as f:
    f.write(f"Transverse field,Frequency,T1z,T1y,T2z,T2y\n")

for freq in f_MAX_list:
    for Hy_t in Hy_list:
        
        sysName=f"P3_{int(T*1e9)}ns_{int(freq*1e-6)}kHz_{int(Hy_t/Oe)}Oe"
        system, region, mesh, alpha = defSys()

        def Hspace_DC(point):
            transverseFieldSize = 0.5e-6
            x, y, z = point
            if (-transverseFieldSize/2 < x < transverseFieldSize/2 and -w/2 < y < w/2 and 0 < z < t):
                return (0, Hy_t, 0)
            else:
                return (0, 0, 0)

        H_transverse = df.Field(mesh, nvdim=3, value=Hspace_DC)
        sysEnergy(Hx,Hy,Hz,H_transverse,system)
        getEq(system)
        directory="images"
        parentDir=f"{sysName}"
        path=os.path.join(parentDir,directory)

        if not os.path.isdir(path):
            os.mkdir(path)
        checkSys(system)
        injectRF(mesh,system,freq)
        if Hy_t==0:
            data=md.Data(sysName) #this contains all the drives up to now. [-1] means the last drive. Check the folder "Py_disk_FMR"
            array=data[-1].to_xarray()
            mz_BL=np.max(array[:, int((3/8*l)/cx), int(w/2/cy), 0, 2])
            my_BL=np.max(array[:, int((3/8*l)/cx), int(w/2/cy), 0, 1])
        m_fft_x, m_fft_y, m_fft_z=dataProcessing()
        getDispersions(m_fft_x, m_fft_y, m_fft_z)
        m_fft_x, m_fft_y, m_fft_z=dataProcessing("left")
        getDispersions(m_fft_x, m_fft_y, m_fft_z,"left")
        mx0,my0,mz0=getTimeEvol("left")
        m_fft_x, m_fft_y, m_fft_z=dataProcessing("right")
        getDispersions(m_fft_x, m_fft_y, m_fft_z,"right")
        getTimeEvol("right")
        m_fft_x, m_fft_y, m_fft_z=dataProcessing("center")
        getDispersions(m_fft_x, m_fft_y, m_fft_z,"center")
        getTimeEvol("center")
        getSpaceEvol(mx0,my0,mz0)
        a,b,c,d=getTramissions((my_BL,mz_BL))
        a = float(a)
        b = float(b)
        c = float(c)
        d = float(d)
        with open(f"./Transmissions.csv",'a') as f:
            f.write(f"{Hy_t},{freq},{a},{b},{c},{d}\n")
      
      
      
        # with open(f"./Transmissions.txt",'a') as f:
        #     f.write(f"Transverse field=:{Hy_t}\n")
        #     f.write(f"frequency={freq}\n")
        #     f.write(f"T1z={a}\n")
        #     f.write(f"T1y={b}\n")
        #     f.write(f"T2z={c}\n")
        #     f.write(f"T2y={d}\n")
        #     f.write("\n\n")
        saveParams(sysName, Hy)
        plt.close('all')