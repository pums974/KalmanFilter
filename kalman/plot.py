import numpy as np
import matplotlib.pyplot as plt

def iter_loadtxt(filename, delimiter="  ", skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


#data = iter_loadtxt("data_old.dat", skiprows=5)
#data = iter_loadtxt("data.dat")
#fig = plt.figure()
#ax = fig.gca()
#ax.set_xlim(np.min(data[:, 0])*0.6, np.max(data[:, 0])*1.1)
#ax.set_ylim(np.min(data[:, 1])*0.8, np.max(data[:, 1])*1.1)
#plt.scatter(data[:, 0],data[:, 1])
#plt.show()

data1 = iter_loadtxt("data_nr4.dat")
data2 = iter_loadtxt("data_nr5.dat")
data3 = iter_loadtxt("data_nr6.dat")
minx = min(np.min(data1[:, 0]), np.min(data2[:, 0]), np.min(data3[:, 0]))
maxx = max(np.max(data1[:, 0]), np.max(data2[:, 0]), np.max(data3[:, 0]))
miny = min(np.min(data1[:, 1]), np.min(data2[:, 1]), np.min(data3[:, 1]))
maxy = max(np.max(data1[:, 1]), np.max(data2[:, 1]), np.max(data3[:, 1]))
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(minx*0.6, maxx*1.1)
ax.set_ylim(miny*0.8, maxy*1.1)
plt.plot(data1[:, 0], data1[:, 1], '.', data2[:, 0], data2[:, 1], '.', data3[:, 0], data3[:, 1], '.')
plt.title('nx=ny=5, dt= 5e-3, nit = 50')
plt.legend(('nr = 4e-3', 'nr = 5e-3', 'nr = 6e-3'))
plt.show()


data1 = iter_loadtxt("data_nit50.dat")
data2 = iter_loadtxt("data_nit100.dat")
data3 = iter_loadtxt("data_nit200.dat")
data4 = iter_loadtxt("data_nit300.dat")
minx = min(np.min(data1[:, 0]), np.min(data2[:, 0]), np.min(data3[:, 0]), np.min(data4[:, 0]))
maxx = max(np.max(data1[:, 0]), np.max(data2[:, 0]), np.max(data3[:, 0]), np.max(data4[:, 0]))
miny = min(np.min(data1[:, 1]), np.min(data2[:, 1]), np.min(data3[:, 1]), np.min(data4[:, 1]))
maxy = max(np.max(data1[:, 1]), np.max(data2[:, 1]), np.max(data3[:, 1]), np.max(data4[:, 1]))
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(minx*0.6, maxx*1.1)
ax.set_ylim(miny*0.8, maxy*1.1)
plt.plot(data1[:, 0], data1[:, 1], '.', data2[:, 0], data2[:, 1], '.', data3[:, 0], data3[:, 1], '.', data4[:, 0], data4[:, 1], '.')
plt.title('nx=ny=5, dt= 5e-3, nr = 3e-3')
plt.legend(('nit = 50', 'nit = 100', 'nit = 200', 'nit = 300'))
plt.show()

data1 = iter_loadtxt("data_dt4.dat")
data2 = iter_loadtxt("data_dt5.dat")
data3 = iter_loadtxt("data_dt6.dat")
minx = min(np.min(data1[:, 0]), np.min(data2[:, 0]), np.min(data3[:, 0]))
maxx = max(np.max(data1[:, 0]), np.max(data2[:, 0]), np.max(data3[:, 0]))
miny = min(np.min(data1[:, 1]), np.min(data2[:, 1]), np.min(data3[:, 1]))
maxy = max(np.max(data1[:, 1]), np.max(data2[:, 1]), np.max(data3[:, 1]))
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(minx*0.6, maxx*1.1)
ax.set_ylim(miny*0.8, maxy*1.1)
plt.plot(data1[:, 0], data1[:, 1], '.', data2[:, 0], data2[:, 1], '.', data3[:, 0], data3[:, 1], '.')
plt.title('nx=ny=5, nr = 3e-3, nit = 50')
plt.legend(('dt = 4e-3', 'dt = 5e-3', 'dt = 6e-3'))
plt.show()


data1 = iter_loadtxt("data_nx4.dat")
data2 = iter_loadtxt("data_nx5.dat")
data3 = iter_loadtxt("data_nx6.dat")
minx = min(np.min(data1[:, 0]), np.min(data2[:, 0]), np.min(data3[:, 0]))
maxx = max(np.max(data1[:, 0]), np.max(data2[:, 0]), np.max(data3[:, 0]))
miny = min(np.min(data1[:, 1]), np.min(data2[:, 1]), np.min(data3[:, 1]))
maxy = max(np.max(data1[:, 1]), np.max(data2[:, 1]), np.max(data3[:, 1]))
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(minx*0.6, maxx*1.1)
ax.set_ylim(miny*0.8, maxy*1.1)
plt.plot(data1[:, 0], data1[:, 1], '.', data2[:, 0], data2[:, 1], '.', data3[:, 0], data3[:, 1], '.')
plt.title('dt = 5e-3, nr = 3e-3, nit = 50')
plt.legend(('nx=ny=4', 'nx=ny=5', 'nx=ny=6'))
plt.show()
