#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:44:48 2020

@author: orynaivashtenko
"""

import numpy as np
import batman
import copy
import os


#ATTENTION, HERE WE GENERATE THE TIME WITH ODD NUMBER OF POINTS!!
#we also generated even, it is enough to choose another time t which is commented
Rs = 1 #Rsun
params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 500.                      #orbital period
params.rp = 0.01/Rs                     #planet radius (in units of stellar radii)
params.a = (params.per/365)**(2/3) * 215/Rs  #semi-major axis (in units of stellar radii)
params.inc = 90.                    #orbital inclination (in degrees)
params.ecc = 0.9                 #eccentricity
params.w = 270.                      #longitude of periastron (in degrees)
params.u = [0.47, 0.23]                #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"       #limb darkening model

global dt_lc #long cadence interval
dt_lc = 1765.5/(60 * 60 * 24) #days long cadence interval
dt_lc = 0.02043368 #almost the same.measured from data
global T_ban
T_ban = 3. #days time of the half-template from the bank. Full time is from -T to T
global L_ban
L_ban = int(T_ban/dt_lc*2) #days time of the bank
global per1
per1 = 50.
global per2
per2 = 500.


def time(T=2*T_ban, dt=dt_lc, Tbins=0, parity='odd', fromzero=False):
    """
    generates time from -T/2 to T/2 (not exactly) with interval dt
    can have t=0 (parity=odd) or not have (parity=even)
    Tbins is number of bins that we want to have with spacing dt. Tbins overwrites T
    ATTENTION Tbins is the overall number, so not (-Tbins, Tbins)!

    Parameters
    ----------
    T : TYPE, optional
        DESCRIPTION. The default is 2*T_ban.
    dt : TYPE, optional
        DESCRIPTION. The default is dt_lc.
    Tbins : TYPE, optional
        DESCRIPTION. The default is 0.
    parity : TYPE, optional
        DESCRIPTION. The default is 'odd'.
    fromzero : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if not fromzero:
        if Tbins==0:
            if parity == 'odd':
                return np.concatenate(((-1)*np.arange(0, T/2, dt)[:0:-1],np.arange(0, T/2, dt)))
            elif parity == 'even':
                return np.concatenate(((-1)*np.arange(dt/2, T/2, dt)[::-1],np.arange(dt/2, T/2, dt)))
            else:
                print('not recognized parity value')
                return -1
        else:
            if type(Tbins)!= int:
                print('Tbins shouls be int')
                return -1
            elif Tbins%2 != 0:
                Tamp = (Tbins+1)//2
                return np.concatenate(((-dt)*np.arange(Tamp)[:0:-1],dt * np.arange(Tamp)))
            else:
                Tamp = Tbins//2
                return np.concatenate(((-dt)*(np.arange(Tamp)[::-1]+0.5), dt*(np.arange(Tamp)+0.5) ))
    else:
        if Tbins==0:
            return np.arange(0, T, dt)
        else:
            if type(Tbins)!= int:
                print('Tbins shouls be int')
                return -1
            else: 
                return dt * np.arange(Tbins)


def transit_shape(t, templ):
    """
    Returns the duration (from the first nonzero to the last nonzero) 
    of the transit, the depth and the width at half-maximum.
    ATTENTION: template is[0,0,...,1,1,...,0,0] and not [1,1,...,0,0,...,1,1] AS BEFORE
    Time array is assumed to be uniformly spaced
    
    Parameters
    ----------
    t : numpy.array 1d of float
        Time array corresponding to the template
    templ : numpy.array 1d of float
        Template.

    Returns
    -------
    dur : float
        Duration of the transit (from the first to the last
        nonzero point) in units of time t.
    dep : float
        Depth of the transit (abs) in units of the given template.
    fwhm : float
        Full width at half maximum in units of time t.

    """
    
    dep = np.abs(np.max(templ) - np.min(templ))
    if dep==0:
        dur = 0
        fwhm = 0
    else:
        ind = np.nonzero(templ)[0]
        if ind[-1]+1>=len(t):
            dur=0
            dep=-1
            fwhm=-1
            print("can't find the end of the flux")
        else:
            dur = t[ind[-1]+1] - t[ind[0]]
            ind = np.where((templ-(np.max(templ)+np.min(templ))/2)>=0)[0]
            if len(ind)==1:
                fwhm=0
            else:
                fwhm = t[ind[-1]] - t[ind[0]]
    return (dur, dep, fwhm)
        


def template_bank(dt=dt_lc, return_parlist=False, return_ordered=True, T=T_ban):
    """
    TEMPLATE BANK
    The bank contains groups of templates:
    
    1) Big eccentricity (0.9), aphelion position, long period (500 days), varying star radii. Covers the longest transits. (10 templates)
    2) Small eccentricity (0), long period (500 days), varying star radii. Covers the long transits. (10 templates)
    3) Earth-Sun with varying periods (7 templates)
    4) Big eccentricities (0.8) in perihelion, average period (350 days). Covers short transits (5 templates)
    5) Big inclinations (everywhere before there was no inclination), long period (500 days). Covers smooth-shaped short transits (5 templates)
    6) A big inclination (everywhere before there was no inclination) and a short period (200 days). A very short smooth-shaped transit (1 template)
    
    Throughout the entire bank, the limb darkening coefficients of the Sun are used


    Parameters
    ----------
    dt : TYPE, optional
        DESCRIPTION. The default is dt_lc.
    return_parlist : TYPE, optional
        DESCRIPTION. The default is False.
    return_ordered : TYPE, optional
        DESCRIPTION. The default is False.
    T : TYPE, optional
        DESCRIPTION. The default is T_ban.

    Returns
    -------
    None.

    """
    # T = T_ban
    # ODD:
    t = np.concatenate(((-1)*np.arange(0, T, dt)[:0:-1],np.arange(0, T, dt))) 
    # EVEN
    # t = np.concatenate(((-1)*np.arange(dt/2, T, dt)[::-1],np.arange(dt/2, T, dt))) 
    # it will not be symmetric, but it will keep the needed step 
    
    # 1. Big eccentricity (0.9), aphelion position, long period (500 days), varying star radii.
    # Covers the longest transits. (10 templates)
    params.per = 500.
    params.ecc = 0.9
    params.w = 270. 
    params.inc = 90.
    r=0.03
    k=4
    RR = 0.3
    Rmin=0.47
    Rss = np.linspace((Rmin-RR)**(1/k),(1.73-RR)**(1/k),10)**k+RR
    templates= np.empty(shape=(0,len(t)), dtype=float)
    parlist=[]
    temps= np.empty(shape=(0,len(t)), dtype=float)
    for R in Rss:
        params.a = (params.per/365)**(2/3) * 215/R
        params.rp = r/R
        m = batman.TransitModel(params, t)    
        flux = m.light_curve(params)
        temps = np.append(temps,[(1-flux)/np.linalg.norm(1-flux)],axis=0)
        parlist.append(1)
    templates = np.append(templates,temps,axis=0)

    # 2. Small eccentricity (0), long period (500 days), varying star radii. 
    # Covers the long transits. (10 templates)
    params.per = 500.
    params.ecc = 0.0
    params.w = 90. 
    params.inc = 90.
    r=0.02
    k=3
    Rss = np.linspace(1**(1/k),1.9**(1/k),10)**k
    temps= np.empty(shape=(0,len(t)), dtype=float)
    for R in Rss:
        params.a = (params.per/365)**(2/3) * 215/R
        params.rp = r/R
        params.rp = r
        m = batman.TransitModel(params, t)    
        flux = m.light_curve(params)
        temps = np.append(temps,[(1-flux)/np.linalg.norm(1-flux)],axis=0)
        parlist.append(2)
    templates = np.append(templates,temps,axis=0)

    # 3. Earth-Sun with varying periods (7 templates)
    params.ecc = 0.0
    params.w = 90. 
    params.inc = 90.
    params.rp = 0.01
    k=3
    pmin=50
    pp=pmin-5
    pmax=350
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),7)**k-(pmin-pp)+pmin
    temps= np.empty(shape=(0,len(t)), dtype=float)
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215
        m = batman.TransitModel(params, t) 
        flux = m.light_curve(params)
        temps = np.append(temps,[(1-flux)/np.linalg.norm(1-flux)],axis=0)
        parlist.append(3)    
    templates = np.append(templates,temps,axis=0)

    # 4. Big eccentricities (0.8) in perihelion, average period (350 days) Covers short transits
    params.ecc=0.8
    params.w=140
    params.rp = 0.03
    k=3
    pmin=30
    pp=0 #pmin-1
    pmax=500
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),5)**k-(pmin-pp)+pmin
    temps= np.empty(shape=(0,len(t)), dtype=float)
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215
        m = batman.TransitModel(params, t) 
        flux = m.light_curve(params)
        temps = np.append(temps,[(1-flux)/np.linalg.norm(1-flux)],axis=0)
        parlist.append(4)    
    templates = np.append(templates,temps,axis=0)

    # 5. Big inclinations (everywhere before there was no inclination), long period (500 days)
    # Covers smooth-shaped short transits
    params.ecc=0.0
    params.w=90.
    r = 0.03
    Rs = 2
    k=3
    pmin=20
    pp=pmin-5
    pmax=700
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),5)**k-(pmin-pp)+pmin
    temps= np.empty(shape=(0,len(t)), dtype=float)
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215/Rs
        params.rp = r/Rs
        params.inc = np.arccos(1/params.a)*180/np.pi
        m = batman.TransitModel(params, t) 
        flux = m.light_curve(params)
        temps = np.append(temps,[(1-flux)/np.linalg.norm(1-flux)],axis=0)
        parlist.append(5) 
    templates = np.append(templates,temps,axis=0)
    
    # 6. A big inclination (everywhere before there was no inclination) and a short period (200 days)
    # A very short smooth-shaped transit    
    params.ecc=0.0
    params.w=270
    params.rp = 0.03
    k=3
    pmin=50
    pp=pmin-5
    pmax=200
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),1)**k-(pmin-pp)+pmin
    temps= np.empty(shape=(0,len(t)), dtype=float)
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215/0.5
        params.rp = 0.03/0.5
        params.inc = np.arccos(1/params.a)*180/np.pi
        m = batman.TransitModel(params, t) 
        flux = m.light_curve(params)
        temps = np.append(temps,[(1-flux)/np.linalg.norm(1-flux)],axis=0)
        parlist.append(6) 
    templates = np.append(templates,temps,axis=0)

    if not return_ordered:
        if not return_parlist:
            return templates 
        else:
            return templates, parlist
    else:
        widths = np.array(list(map(lambda tem: transit_shape(t, tem)[0], templates)))
        order = np.argsort(widths)
        templates=templates[order]
        return templates
        # parlist=np.array(parlist)[order]
        # widths = np.array(list(map(lambda tem: transit_shape(t, tem)[0], templates)))
        

def template_bank_subbin_shift(dt=dt_lc, dP=None, T_mission=1400., Pmin=per1, 
                               n_points_int_min=20, return_ordered=True,
                               shifts_num=3, return_shifts=True):
    """
    The number of needed shifts is determined so that the inner product of the 
    shifted templates >0.9, but <0.995

    Parameters
    ----------
    dt : float
        Time step of the time array in days.
    dP : float
        Time step in the measurable periods list. I None, will be computed as the
        resolved period step from the total data duration T_mission and the minimal
        available period Pmin. (It will also be adjusted to divide dt into integer number)
    T_mission : float
        Total duration of the data
    Pmin : float
        Minimal period in the search. By default is per1 of template_module
    n_points_int_min : TYPE, optional
        Minimal number of initial points for one point after integrative interpolation
    return_ordered : bool
        whether to return the template bank ordeted by increasing transit duration
    shifts_num : None or array 1d of int
        Set manually how many shifts will be for each template.
        The length should be the same as the lentgh of template bank
    return_shifts : bool
        Whether to return shifts array

    Returns
    -------
    templates_shifts, list of lists of numpy.arrays
        Each of the sublists is the list of all the shifts for the
        corresponding template. 

    """   
    if dP is None:
        dP = dt*Pmin/T_mission
        dP = dt/(np.round(dt/dP)) #round dP to split time step in equal intervals for making shifts
    nn = max(int(np.ceil(n_points_int_min*dP/dt)), 1) #n_points_int_initial = int(dt/dP) -> after fine-graining n_points_int = dt*nn/dP
    dt_int0 = dP/nn #time step in the time sample used for integration
    dt_int = dt_int0
    tb_int0 = template_bank(dt=dt_int, return_ordered=True)
    t_int0 = time(dt=dt_int)
    t_int = t_int0
    tb = template_bank(dt=dt, return_ordered=True)
    t = time(dt=dt)
    
    if shifts_num is not None:
        if type(shifts_num)!=int:
            assert len(shifts_num)==len(tb)
    else:
        n_shifts_P = int(np.round(dt/dP)) # partition corresponding to the period resolution
                                      # (maximal partition that we consider)
    
    templates_shifts = []
    shifts_shifts = []
    for i in range(len(tb)):
        dt_int = dt_int0 #it will be adjusted for each template to split dt into equal parts
        template_int = np.array(tb_int0[i])
        t_int = t_int0
        shifts=None
        if ((type(shifts_num)!=int) and (shifts_num[i]==1) or (type(shifts_num)==int) and (shifts_num==1)):
            template_shifts = [tb[i]]
            shifts = [0]        
        elif tb[i]@np.roll(tb[i],1)>=0.95: #is the shifted by dt template is close enough, no need to put sub-bin shifts
            template_shifts = [tb[i]]
            shifts = [0]
        else:
            if shifts_num is not None:
                if type(shifts_num)==int:
                    dt_int = dt/(np.ceil(n_points_int_min/shifts_num)*shifts_num)
                    shifts = np.arange(shifts_num)*int(np.ceil(n_points_int_min/shifts_num))
                    template_int = single_template(i, dt=dt_int, ordered_bank=True) #use new dt_int
                    t_int = time(dt=dt_int)
                else:
                    dt_int = dt/(np.ceil(n_points_int_min/shifts_num[i])*shifts_num[i])
                    shifts = np.arange(shifts_num[i])*int(np.ceil(n_points_int_min/shifts_num[i]))
                    template_int = single_template(i, dt=dt_int, ordered_bank=True) #use new dt_int
                    t_int = time(dt=dt_int)
            else:
                shifts_done = False
                for k in np.arange(2, n_shifts_P): # fine-graining the partition and checking whether the templates get close enough
                    if template_int@np.roll(template_int,int(dt/k/dt_int))>=0.95: # if densely sampled templates ate too far, just take smaller partition  
                        temp_shif1 = integ_interp(t, t_int, tb_int0[i])[1]
                        temp_shif1 = temp_shif1/np.linalg.norm(temp_shif1)
                        temp_shif2 = integ_interp(t, t_int, np.roll(tb_int0[i],int(dt/k/dt_int)))[1]
                        temp_shif2 = temp_shif2/np.linalg.norm(temp_shif2)
                        if temp_shif1@temp_shif2>=0.95: #check thet the integ-interpolated templates will be close enough
                            dt_int = dt/(np.ceil(n_points_int_min/k)*k)
                            shifts = np.arange(k)*int(np.ceil(n_points_int_min/k))
                            template_int = single_template(i, dt=dt_int, ordered_bank=True) #use new dt_int
                            t_int = time(dt=dt_int)
                            shifts_done = True
                            break
                if not shifts_done:
                    shifts = np.arange(0,n_shifts_P)*int(dP/dt_int)
            t_int = np.concatenate(([t_int[0]-dt_int], t_int, [t_int[-1]+dt_int])) #add one extra bin for integ-interpolation purposes
            template_int = np.concatenate(([0.], template_int, [0.])) #add one extra bin for integ-interpolation purposes
            template_shifts = []
            for s in range(len(shifts)):
                temp_shif = integ_interp(t, t_int, np.roll(template_int,shifts[s]))[1]
                temp_shif = temp_shif/np.linalg.norm(temp_shif)
                template_shifts.append(temp_shif)
        templates_shifts.append(template_shifts)
        shifts_shifts.append(np.array(shifts) * dt_int)
    if return_shifts:
        return templates_shifts, shifts_shifts
    else:
        return templates_shifts


def template_bank_parameters(return_ordered=True, dt=dt_lc, T=T_ban):  
    """
    Generate the light curve parameters that were used in the template bank
    (without generating actual templates in order not to spend time.)
    Parameters are the object of the class batman.TransitParams().
    The order of the parameters in the list is the same as of the templates
    in template_bank().

    Parameters
    ----------
    return_ordered : bool
        Whether to sort the list by increasing width of the transits generated
        from these parameters
    dt : float
        Time step of the sampling of the template. Used only in case that 
        need to order the parameters (return_ordered=True), because for 
        that an actual template bank is produced.
    T : float
        Half total time, also used only if return_ordered. The default is T_ban.

    Returns
    -------
    parameter_list : list of oblects batman.TransitParams()
        Parameters used to generate the template bank.
        Needed for single templates generation.

    """
    parameter_list = []
    params.per = 500.
    params.ecc = 0.9
    params.w = 270. 
    params.inc = 90.
    r=0.03
    k=4
    RR = 0.3
    Rmin=0.47
    Rss = np.linspace((Rmin-RR)**(1/k),(1.73-RR)**(1/k),10)**k+RR
    for R in Rss:
        params.a = (params.per/365)**(2/3) * 215/R
        params.rp = r/R
        parameter_list.append(copy.deepcopy(params))

    # 2. Small eccentricity (0), long period (500 days), varying star radii. 
    # Covers the long transits. (10 templates)
    params.per = 500.
    params.ecc = 0.0
    params.w = 90. 
    params.inc = 90.
    r=0.02
    k=3
    Rss = np.linspace(1**(1/k),1.9**(1/k),10)**k
    for R in Rss:
        params.a = (params.per/365)**(2/3) * 215/R
        params.rp = r/R
        params.rp = r
        parameter_list.append(copy.deepcopy(params))
        
    # 3. Earth-Sun with varying periods (7 templates)
    params.ecc = 0.0
    params.w = 90. 
    params.inc = 90.
    params.rp = 0.01
    k=3
    pmin=50
    pp=pmin-5
    pmax=350
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),7)**k-(pmin-pp)+pmin
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215
        parameter_list.append(copy.deepcopy(params))
        
    # 4. Big eccentricities (0.8) in perihelion, average period (350 days) Covers short transits
    params.ecc=0.8
    params.w=140
    params.rp = 0.03
    k=3
    pmin=30
    pp=0 #pmin-1
    pmax=500
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),5)**k-(pmin-pp)+pmin
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215
        parameter_list.append(copy.deepcopy(params))

    # 5. Big inclinations (everywhere before there was no inclination), long period (500 days)
    # Covers smooth-shaped short transits
    params.ecc=0.0
    params.w=90.
    r = 0.03
    Rs = 2
    k=3
    pmin=20
    pp=pmin-5
    pmax=700
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),5)**k-(pmin-pp)+pmin
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215/Rs
        params.rp = r/Rs
        params.inc = np.arccos(1/params.a)*180/np.pi
        parameter_list.append(copy.deepcopy(params))
    
    # 6. A big inclination (everywhere before there was no inclination) and a short period (200 days)
    # A very short smooth-shaped transit    
    params.ecc=0.0
    params.w=270
    params.rp = 0.03
    k=3
    pmin=50
    pp=pmin-5
    pmax=200
    Ps = np.linspace((pmin-pp)**(1/k),(pmax-pp)**(1/k),1)**k-(pmin-pp)+pmin
    for p in Ps:
        params.per = p
        params.a = (p/365)**(2/3) * 215/0.5
        params.rp = 0.03/0.5
        params.inc = np.arccos(1/params.a)*180/np.pi
        parameter_list.append(copy.deepcopy(params))

    if not return_ordered:
        return parameter_list
    else:
        templates = template_bank(dt=dt_lc, return_parlist=False, return_ordered=False, T=T_ban)
        t = np.concatenate(((-1)*np.arange(0, T_ban, dt_lc)[:0:-1],np.arange(0, T_ban, dt_lc)))
        widths = np.array(list(map(lambda tem: transit_shape(t, tem)[0], templates)))
        order = np.argsort(widths)
        parameter_list=[parameter_list[i] for i in order]
        return parameter_list


template_bank_parameter_list = template_bank_parameters(dt=dt_lc, return_ordered=True, T=T_ban)


def single_template(template_ind, dt=dt_lc, ordered_bank=True, T=T_ban):
    """
    To create only one template from the template bank and not to dpend time on others

    Parameters
    ----------
    template_ind : int
        Index of the template from the ordered bank if ordered_bank==True,
        and from the non-ordered bank otherwise
    dt : float
        Time stap of sampling
    ordered_bank : bool
        Whether to order the templates by increasing width
    T : float
        Half total time of the template bank.

    Returns
    -------
    numpy.array
        Template from the template bank.

    """
    if ordered_bank:
        global template_bank_parameter_list
    else:
        template_bank_parameter_list = template_bank_parameters(dt=dt_lc, return_parlist=False, return_ordered=ordered_bank, T=T)
    t = np.concatenate(((-1)*np.arange(0, T, dt)[:0:-1],np.arange(0, T, dt))) 
    m = batman.TransitModel(template_bank_parameter_list[template_ind], t) 
    flux = m.light_curve(template_bank_parameter_list[template_ind])
    return (1-flux)/np.linalg.norm(1-flux)
    
    
def integ_interp(x1, x0, y0):
    """
    Interpolates the signal (making it more coarse-grained) using integration.
    Each output point is the mean of the input array located at a distance less than
    half-bin of the coarse array.
    -----------------------
    x1 : np.array or list or float. Output arguments array
    If array or list, has to be equally spaced.
        If int, will be treated as step dx1 in the new array.
        Number of points will be chosen automatically to cover the input.
        Integration will be performed in such way that these points are 
        in the middle of integration domain
    x0 : np.array or list. Input arguments array
        Has to be equally spaced
    y0 : np.array or list. Input function values array
    """
    if type(x1)!=int:
        assert type(x1)==np.ndarray, 'Incorrect type of the new x (x1) array!'
        assert np.isclose(np.std((x1-np.roll(x1,-1))[:-1]),0), 'New array of arguments (x1) has to be equally spaced!'
    if type(x1)==np.ndarray or type(x1)==list:
        dx1 = x1[1] - x1[0]
        dx0 = x0[1] - x0[0]
        assert x0[0] <= x1[0] - dx1/2, 'Sample too short at the beginning!'
        assert x0[-1] >= x1[-1] + dx1/2, 'Sample too short at the end!'
        assert dx0 <= dx1, 'Data points not fine enough to integrate!'
    else: 
        dx1 = x1
        assert dx0 <= dx1, 'Data points not fine enough to integrate!'
        x1 = np.arange(x0[0] + dx1/2,x0[-1] - dx1/2, dx1)
    xint = np.append(x1, x1[-1] + dx1) - dx1/2 #array used for integration (shifted by half-bin and extended)
    y1 = np.zeros(len(x1))*np.nan
    for i in range(len(x1)):
        i1 = np.where(x0>xint[i])[0][0]  # ATTENTION: BOUNDARY POINTS WILL BE IGNORED 
        i2 = np.where(x0<xint[i+1])[0][-1]
        y1[i] = np.nanmean(y0[i1:i2+1])
    return x1, y1


def durations(t=None, templates=None):
    if t is None and templates is None:
        t = time()
        templates = template_bank()
    return np.array(list(map(lambda tem: transit_shape(t, tem)[0], templates)))        

def depths(t=None, templates=None):
    if t is None and templates is None:
        t = time()
        templates = template_bank()
    return np.array(list(map(lambda tem: transit_shape(t, tem)[1], templates)))

def fwhms(t=None, templates=None):
    if t is None and templates is None:
        t = time()
        templates = template_bank()
    return np.array(list(map(lambda tem: transit_shape(t, tem)[2], templates))) 

def transit_lengths(templates=None):
    if templates is None:
        templates = template_bank()
    return np.array(list(map(lambda tem: len(np.nonzero(tem)[0]), templates)))        


def normalize(x, t=0, bymax=False, bysum=False):
    """
    Normalizes the vector with following options:
    1) np.linalg.norm
    2) by the maximum (bymax=True)
    3) by the sum (bysum=True)
    4) can perform sum and integrals with respect to time axis (if t specified))

    Parameters
    ----------
    x : np.array
        a vector to normalize.
    t : float or np.array, optional
        If t is array, it is time series.
        If t is float it is time step dt.
        if specified, normalization is performed by time integral 
        (squared or not squared if bysum). The default is 0.
    bymax : bool, optional
        if True, normalization is done by maximum. The default is False.
    bysum : bool, optional
        if True, normalization is done by sum. The default is False.

    Returns
    -------
    np.array
        the normalized vector.

    """
    if bymax:
        return x/np.max(x)
    elif bysum:
        if type(t)==float or type(t)==int:
            if t==0:
                return x/np.abs(np.sum(x))
            else:
                return x/np.abs(np.sum(t*x))
        else:
            dt = (np.roll(t,-1)-t)[:-1]
            dt = np.append(dt, dt[-1])
            return x/np.abs(np.sum(dt*x))
    else:
        if type(t)==float or type(t)==int:
            if t==0:
                return x/np.linalg.norm(x)
            else:
                return x/np.linalg.norm(t*x)
        else:
            dt = (np.roll(t,-1)-t)[:-1]
            dt = np.append(dt, dt[-1])
            return x/np.linalg.norm(dt*x)


def flux_par(t=None, params=None, per=365., rp=0.01, a=215., inc=90., ecc=0., w=90., u1=0.47, u2=0.23, t0=0,
             supersample_factor=None, exp_time=None):
    """
    

    Parameters
    ----------
    t : np.array
        time, array (not necessarily evenly spaced), any units consistent with period
    per : float, optional
        period, any units consistent with time. The default is 365..
    rp : float, optional
        planet radius in units of stellar radius. The default is 0.01.
    a : float, optional
        major semi-axis, in units of stellar radius. The default is 215..
    inc : float, optional
        inclination in degrees. The default is 90., it means there is no inclination, a straight transit.
    ecc : float, optional
        eccentricity. The default is 0..
    w : float, optional
        periapsis argument, degraee. The default is 90., which is perihelion towards us
    u1 : float, optional
        first limb darkening coefficient for the quadratic model, in [0,1]. The default is 0.47.
    u2 : float, optional
        second limb darkening coefficient for the quadratic model, in [0,1]. The default is 0.23.
    t0 : float, optional
        time of inferior conjunction. The default is 0.

    Returns
    -------
    flux : np.array
        flux (periodic) at corresponding times with max at 1 and dimmings on transits.
    #returns the flux as function of given physical parameters

    """
    if t is None:
        t = time()
    if params is None:
        params = batman.TransitParams()
        params.t0 = t0                       #time of inferior conjunction
        params.per = per                      #orbital period
        params.rp = rp                     #planet radius (in units of stellar radii)
        params.a = a                       #semi-major axis (in units of stellar radii)
        params.inc = inc                     #orbital inclination (in degrees)
        params.ecc = ecc                      #eccentricity
        params.w = w                       #longitude of periastron (in degrees)
        params.u = [u1, u2]                #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"       #limb darkening model
    if (supersample_factor is not None) and (exp_time is not None):
        m = batman.TransitModel(params, t, supersample_factor = supersample_factor, exp_time = exp_time)
    else:
        m = batman.TransitModel(params, t)    #initializes model
    flux = m.light_curve(params)          #calculates light curve
    return flux


def write_template_bank_to_file(name='template_bank', var=None, location=None):
    """
    Prints array (list or numpy.array of float) variable var to file named name in the location location.
    If the var is not given, the name of the variable is the same as name

    Parameters
    ----------
    name : str
        Name of the file ('.txt' is not needed). If var is None, it is also 
        the name of the variable to be printed to the file.
    var : list or array
        Variable to be printed to the file. If var is None, name is the name 
        of the variable to be printed to the file.
    location : str
        Location where to create the file. If None, the current local directory
        will be used

    Returns
    -------
    None.

    """
    if name[-4:]=='.txt':
        name = name[:-4]
    if var==None:
        var = template_bank_subbin_shift(dt=dt_lc, dP=None, T_mission=1400., Pmin=per1, 
                               n_points_int_min=50, return_ordered=True, shifts_num=3,
                               return_shifts=False)
    if location is None:
        location = os.getcwd() 
        fileloc = os.path.join(location, name)+'.txt'
    else:
        fileloc = name +'.txt'
    file = open(fileloc, 'w')
    print(*var, file=file)
    file.close()


#Function for plotting with smoothly changing colors from red to blue
#s is the smoothness parameter
#returns the dictionary of your varying parameter and the color
def colordict(x, s=0):
    m = min(x)
    M = max(x)
    if s==0:
        s=(M-m)/3.5  
    return {xx : (round(np.exp(-((xx-m)-(m+M)/10 )**2/(s**2)),3),
                  round(np.exp(-(xx-(m+M)/2.1)**2/(2.1*s**2))/1.4,3),
                  round(np.exp(-((xx-M)+(m+M)/10 )**2/(2.*s**2)),3)) for xx in x}

