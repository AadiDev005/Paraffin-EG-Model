import numpy as np

def shift(array, shift_x, shift_y):
    shifted_array=np.copy(array)

    # Shift along the cols
    if shift_x>0:
        shifted_array[shift_x:, :]=array[:-shift_x, :]
        shifted_array[:shift_x, :]=0
    elif shift_x<0:
        shifted_array[:shift_x, :]=array[-shift_x:, :]
        shifted_array[shift_x:, :]=0

    # Shift along the rows
    if shift_y>0:
        shifted_array[:, shift_y:]=shifted_array[:, :-shift_y]
        shifted_array[:, :shift_y]=0
    elif shift_y<0:
        shifted_array[:, :shift_y]=shifted_array[:, -shift_y:]
        shifted_array[:, shift_y:]=0
    return shifted_array

def lbm(grid, paraffinRelaxationTime=0.59567, saturatedEGRelaxationTime=1.2764):
    if type(grid) is list:
        grid=np.array(grid)
    rows, cols=grid.shape
    tHot=293
    tCold=283
    parRelaxTime=paraffinRelaxationTime
    satEGRelaxTime=saturatedEGRelaxationTime
    dirs=[[1,0],[0,1],[-1,0],[0,-1]]
    T=np.full((rows, cols), (tHot+tCold)/2)
    tauMap=np.where(grid==1, parRelaxTime, satEGRelaxTime)
    T[:,0]=tHot
    T[:,-1]=tCold
    omega=1/4
    distF=np.zeros((4, rows, cols))
    distF[:,:,:]=T[:,:]*omega
    tolerance=1e-8
    for i in range(1000000):
        T_old=T.copy()

        T=np.sum(distF, axis=0)

        # collision
        for dir in range(4):
            distF[dir]-=(distF[dir]-T*omega)/tauMap
                
        # streaming step
        for dir, [x,y] in enumerate(dirs):
            distF[dir]=shift(distF[dir], x, y)

        # top/bottom neumann boundary conditions
        distF[0,0,:]=distF[2,0,:]
        distF[2,-1,:]=distF[0,-1,:]

        # reset left/right boundary conditions
        distF[1,:,0]=tHot-distF[0,:,0]-distF[2,:,0]-distF[3,:,0]
        distF[3,:,-1]=tCold-distF[0,:,-1]-distF[1,:,-1]-distF[2,:,-1]

        # update temp
        T=np.sum(distF, axis=0)

        # verify how much temp changed 
        error=np.sqrt(np.sum(np.square((T-T_old)))/np.sum(np.square(T)))
        if error<tolerance:
            print("CONVERGED ON STEP", i, "WITH ERROR", error)
            break
    deltaT=tHot-tCold
    qFlux=(np.sum(distF[1,:,:])-np.sum(distF[3,:,:]))*(np.mean(tauMap)-0.5)/np.mean(tauMap)
    cond=1000*qFlux/(deltaT*rows*cols)
    return cond, T