import numpy as np
import sympy
import scipy.io as sio
v = sympy.symbols('v')


def vdw(tr, v):
    p = 8 * tr / (3 * v - 1) - 3 / v / v
    return p


def maxwell_equal_area(tr):
    for v1 in np.linspace(0.4,1,500):
        s = sympy.solve(vdw(tr,v) - vdw(tr,v1), v)
        if sympy.im(s[0])<1e-5 and sympy.im(s[1])<1e-5 and sympy.im(s[2])<1e-5:
            print('起始点v=%s' % v1)
            for v2 in np.linspace(v1,1,100000):
                s = sympy.solve(vdw(tr,v) - vdw(tr,v2), v)
                dp = sympy.integrate(vdw(tr,v2) - vdw(tr,v),(v,sympy.re(s[0]),sympy.re(s[2])))
                print('dp=%s' % (dp))
                if np.abs(dp)<1*10**-3:
                    vliquid, vgas = s[0],s[2]
                    return sympy.re(vliquid), sympy.re(vgas)

v = sympy.symbols('v')
num = 0
data={}
for tr in np.linspace(0.5,1,20):
    vliquid, vgas = maxwell_equal_area(tr)
    num = num + 1
    print('tr=%s,vl=%s,vg=%s'%(tr,vliquid,vgas))
    data[num] = [tr, vliquid, vgas]
    sio.savemat('data',{'data':data})



