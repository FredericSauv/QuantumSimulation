# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:39:02 2017

@author: fs
"""

import matplotlib.pylab as plt
import numpy as np

import sys
sys.path.insert(0, './../QuantumSimulation')

import QuantumSimulation.Simulation.Simulation as sim
import QuantumSimulation.Models.ToyModel as toy



#       ======================================     #
#                    Exploit simulations          #
#                To generate results/plots
#       ======================================     #
class Results:
    def __init__(self, listSimuls, names='', STF = 3, LTF = 25, lag = 25): 
        
        #retrieve main info from each simul contained in listSimuls and store them as list
        self.simulName = names
        self.nbSim = len(listSimuls)
        self.listSimuls = listSimuls
        self.listcTypes = [s.cType for s in listSimuls]
        self.listTarget = [s.cTarg for s in listSimuls]
        self.listdt = [s.p.dt for s in listSimuls]
        self.listTimeArray = [s.model.tArray for s in listSimuls]
        self.listOm = [s.p.omega for s in listSimuls]
        self.listOmEff = [self.getFreqEff(s) for s in listSimuls]
        self.listOmEffScaled = [self.getFreqEff(s) / (s.p.omega * s.p.n_int) for s in listSimuls]
        self.listTEff = [2* np.pi / om for om in self.listOmEff]
        self.listtmax = [s.p.tmax for s in listSimuls]
        self.listnTeff = [int(np.floor(self.listtmax[i] / self.listTEff[i]))  for i in np.arange(self.nbSim)]
        
        # Parameters used for comparison
        self.STF, self.LTF, self.lag = STF, LTF, lag

        # Charte graphique (color/pointers scheme)
        self.CS_0 = ['-b', '-r', '-g', '-k']
        self.CS_1 = ['-bo', '-rs', '-gv', '-k^']
        self.CS_2 = ['--bo', '--rs', '-gv', '-k^']
        self.CS_3 = ['--b', '--r', '--g', '--k']

        #
        self.genAllResults(STF, LTF, lag)


    def genAllResults(self, STF, LTF, lag):
        self.genAllTimeFrames(STF, LTF, lag)
        self.genAllFidelity()
        self.genAllHeating()

# Functions to generate results (gen = compute and store in the Results object)
    def genAllTimeFrames(self, STF, LTF, later ):
        """
            For all simul gen and store relevant Time Frames (as slices)
            STF (short time frame)/ LTF (long time frame)  + strobo
        """
        dico1 = {}
        dico2 = {}
        listSimuls = self.listSimuls
        nbSim = self.nbSim

        dico1['Short'] = [self.getTimeFrames(s, lengthTF = STF) for s in listSimuls]
        dico1['Long'] = [self.getTimeFrames(s, lengthTF = LTF) for s in listSimuls]
        dico2['Short'] = [self.getTimeFrames(s, lengthTF = STF, strobo = True) for s in listSimuls]
        dico2['Long'] = [self.getTimeFrames(s, lengthTF = LTF, strobo = True) for s in listSimuls]
        
        # For plotting dynamics over 1 period
        dico1['1TF'] =  [self.getTimeFrames(s, lengthTF = 1) for s in listSimuls]
        dico1['1TFlag'] = [self.getTimeFrames(s, laterTime = later,lengthTF = 1) for s in listSimuls]
        dico1['1TFlast'] = [self.getTimeFrames(listSimuls[i],laterTime = self.listnTeff[i]-1, lengthTF = 1) for i in np.arange(nbSim)]
        dico1['1TFmid'] = [self.getTimeFrames(listSimuls[i], laterTime = int(self.listnTeff[i]/2), lengthTF = 1) for i in np.arange(nbSim)]

        dico2['1TF'] = [self.getTimeFrames(s, lengthTF = 1, strobo = True) for s in listSimuls]
        dico2['1TFlag'] = [self.getTimeFrames(s, laterTime = later, lengthTF = 1, strobo = True) for s in listSimuls]
        dico2['1TFlast'] = [self.getTimeFrames(listSimuls[i],laterTime = self.listnTeff[i]-1, lengthTF = 1, strobo = True) for i in np.arange(nbSim)]
        dico2['1TFmid'] = [self.getTimeFrames(listSimuls[i], laterTime = int(self.listnTeff[i]/2), lengthTF = 1, strobo = True) for i in np.arange(nbSim)]
        
        self.TF = dico1
        self.TFstro = dico2
        #self.listSlice    

    def genAllFidelity(self, fidChannel = 'FME_K_FNS'):
        """
            Generate (and store) fidelilites: all the simuls x 2 TF (short or long) x strobo (or not) 
        """
        slSTF = self.TF['Short']
        slLTF = self.TF['Long']
        slSTFstr = self.TFstro['Short']
        slLTFstr = self.TFstro['Long']

        dico1 = {}
        dico2 = {}

        dico1['ShortTF'] = np.array([self.getAvgFidelity(self.listSimuls[ind], tslice = slSTF[ind], fidChannel = 'FME_K_FNS') for ind in np.arange(self.nbSim)])
        dico1['LongTF'] = np.array([self.getAvgFidelity(self.listSimuls[ind], tslice = slLTF[ind], fidChannel = 'FME_K_FNS') for ind in np.arange(self.nbSim)])
        dico2['ShortTF'] = np.array([self.getAvgFidelity(self.listSimuls[ind], tslice = slSTFstr[ind], fidChannel = 'FME_K_FNS') for ind in np.arange(self.nbSim)])
        dico2['LongTF'] = np.array([self.getAvgFidelity(self.listSimuls[ind], tslice = slLTFstr[ind], fidChannel = 'FME_K_FNS') for ind in np.arange(self.nbSim)])

        self.fid = dico1 
        self.fid_strobo = dico2

    def genAllHeating(self, tslice = None):
        """
        generate (and store heating results ) for all simul x 2 TimeFrames x strobo (or not)
        """
        slSTF = self.TF['Short']
        slLTF = self.TF['Long']
        slSTFstr = self.TFstro['Short']
        slLTFstr = self.TFstro['Long']

        dico1 = {}
        dico2 = {}

        dico1['ShortTF'] = np.array([self.getHeating(self.listSimuls[ind], tslice = slSTF[ind]) for ind in np.arange(self.nbSim)])
        dico1['LongTF'] = np.array([self.getHeating(self.listSimuls[ind], tslice = slLTF[ind]) for ind in np.arange(self.nbSim)])
        dico2['ShortTF'] = np.array([self.getHeating(self.listSimuls[ind], tslice = slSTFstr[ind]) for ind in np.arange(self.nbSim)])
        dico2['LongTF'] = np.array([self.getHeating(self.listSimuls[ind], tslice = slLTFstr[ind]) for ind in np.arange(self.nbSim)])

        self.heat = dico1 
        self.heat_strobo = dico2

    def genAllDynamics(self, tslice = None):
        """
        dynamics selected at specific time for the purpose of plotting
        """
        # population (0, g, +):
        #dico = {}
        #dico[] =  [self.listSimuls[i] for i in np.arange(nbSim)]

        # heating

        pass


# ========================== #
# Presentation of the simuls (either for a list or for several lists to be compared)
#     PlotCompareFidelities
#     PlotCompareHeating
#     PlotEvolution (maybe not here)
# ========================== #
    def plotCompareFidelity(self, xSort = 'omeff', nameGraph='', *args):
            """
                *args expected (if any) listSimulExtra1, nameExtra1, listSimulExtra2, nameExtra2 
            """
            assert ((len(args) % 2)== 0), 'plotCompareFidelity: pb in the args provided'
            nbExtraRes = int(len(args)/2)
            nbRes = 1 + nbExtraRes
            print(len(args))
       
            yAxis, yAxisSTF, yAxisStrobo, yAxisSTFStrobo, labels, xAxis = [], [], [], [], [], []
            ymin1, ymin2 = [], []

            # First list of simuls
            if (xSort == 'omeff'):
                xAxis.append(self.listOmEffScaled)
                xLabel = '$Amplitude(\omega_0)$'
            else:
                pass     
            yAxis.append(100 * self.fid['LongTF'])
            yAxisSTF.append(100 * self.fid['ShortTF'])    
            yAxisStrobo.append(100 * self.fid_strobo['LongTF'])
            yAxisSTFStrobo.append(100 * self.fid_strobo['ShortTF'])
            labels.append(nameGraph)
            ymin1.append(np.min(yAxisSTF))
            ymin2.append(np.min(yAxisSTFStrobo))
            
            if(len(args)>0):
                for i in np.arange(nbExtraRes):
                    simTmp = args[(2*i)]
                    xAxis.append(simTmp.listOmEffScaled)
                    yAxis.append(100 * simTmp.fid['LongTF'])
                    yAxisSTF.append(100 * simTmp.fid['ShortTF'])
                    yAxisStrobo.append(100 * simTmp.fid_strobo['LongTF'])
                    yAxisSTFStrobo.append(100 * simTmp.fid_strobo['ShortTF'])
                    ymin1.append(np.min(yAxisSTF))
                    ymin2.append(np.min(yAxisSTFStrobo))
                    labels.append(args[(2*i + 1)])
    
            ymin1 = max(0,np.min(np.array(ymin1))) - 0.01
            ymin2 = max(0,np.min(np.array(ymin2))) - 0.01
    
            #First Graph All time (i.e. not nonly strobo)
            title = nameGraph + ' all time'
            fig, ax = plt.subplots(1)
            # Inplot
            ax2 = fig.add_axes([0.2, 0.2, 0.25, 0.25])
            ax2.tick_params(labelsize=8)
            ax2.set_ylim([ymin1, 100.01])
            ax2.set_title('Short time frame', fontsize=10)
            ax.set_xlabel(xLabel)
            ax.set_ylabel('Fidelity (%)')
            ax.set_ylim([-0.01, 100.01])
            ax.grid()
            ax.set_title(title)
    
            for i in np.arange(nbRes):
                ax.plot(xAxis[i], yAxis[i], self.CS_1[i], label = labels[i])
                ax2.plot(xAxis[i], yAxisSTF[i],self.CS_2[i],markerfacecolor='none', label = labels[i])
            if (nbRes>1):
                ax.legend(loc='best')

            #ax2.plot(xAxis2, r2f_STF,'--rs',markerfacecolor='none', label='2harmonics - short time frame')
        
    
            #Second Graph Strobo result
            title = nameGraph + ' strobo'
            fig, ax = plt.subplots(1)
            # Inplot
            ax2 = fig.add_axes([0.2, 0.2, 0.25, 0.25])
            ax2.tick_params(labelsize=8)
            ax2.set_ylim([ymin2, 100.01])
            ax2.set_title('Short time frame', fontsize=10)
            ax.set_xlabel(xLabel)
            ax.set_ylabel('Fidelity (%)')
            ax.set_ylim([-0.01, 100.01])
            ax.grid()
            ax.set_title(title)
    
            for i in np.arange(nbRes):
                ax.plot(xAxis[i], yAxisStrobo[i], self.CS_1[i], label = labels[i])
                ax2.plot(xAxis[i], yAxisSTFStrobo[i],self.CS_2[i],markerfacecolor='none', label = labels[i])
            if (nbRes>1):
                ax.legend(loc='best')
    
        
    
    def plotCompareHeating(self, xSort = 'omeff',  inset = True, insetPos = None, nameGraph='',  *args):
        # PREPARE DATA
            assert ((len(args) % 2)== 0), 'plotCompareFidelity: pb in the args provided'
            nbExtraRes = int(len(args)/2)
            nbRes = 1 + nbExtraRes
            
            if (insetPos is None):
                insetPos = [0.49, 0.21, 0.4, 0.30]
            xAxis = []
            if (xSort == 'omeff'):
                xLabel = '$Amplitude (\omega_0)$'
                xAxis.append(self.listOmEffScaled)
                for i in np.arange(nbExtraRes):
                    xAxis.append(args[(2*i)].listOmEffScaled)
            else:
                xLabel = 'patati'
                pass    
            print(xLabel)
            yAxis, yAxisSTF, yAxisStrobo, yAxisSTFStrobo, labels = [], [], [], [], []
            ymax1, ymax2 = [], []
            ymin1, ymin2 = [], []
            
            yAxis.append(np.log10(self.heat['LongTF']))
            yAxisSTF.append(np.log10(self.heat['ShortTF']))
            yAxisStrobo.append(np.log10(self.heat_strobo['LongTF']))
            yAxisSTFStrobo.append(np.log10(self.heat_strobo['ShortTF']))
            labels.append(nameGraph)
            ymin2.append(np.floor(np.min(yAxisSTF[-1])))
            ymax2.append(np.ceil(np.max(yAxis[-1])))
            ymin1.append(np.floor(np.min(yAxisSTFStrobo[-1])))
            ymax1.append(np.ceil(np.max(yAxisStrobo[-1])))
            print(nbExtraRes)
            if(len(args)>0):
                for i in np.arange(nbExtraRes):
                    simTmp = args[(2*i)]
                    yAxis.append(np.log10(simTmp.heat['LongTF']))
                    yAxisSTF.append(np.log10(simTmp.heat['ShortTF']))
                    yAxisStrobo.append(np.log10(simTmp.heat_strobo['LongTF']))
                    yAxisSTFStrobo.append(np.log10(simTmp.heat_strobo['ShortTF']))
                    labels.append(args[2*i + 1])
                    ymin2.append(np.floor(np.min(yAxisSTF[-1])))
                    ymax2.append(np.ceil(np.max(yAxis[-1])))
                    ymin1.append(np.floor(np.min(yAxisSTFStrobo[-1])))
                    ymax1.append(np.ceil(np.max(yAxisStrobo[-1])))
                    
    
            ymin1 = np.min(np.array(ymin1))
            ymax1 = np.max(np.array(ymax1))
            ymin2 = np.min(np.array(ymin2))
            ymax2 = np.max(np.array(ymax2))
            
            print(ymin1, ymax1, ymin2, ymax2)
    
            #GRAPH
            # Main heat at strobo times (straight) LTF (dotted) STF
            title = nameGraph + ' Stroboscopic times'
            fig, ax = plt.subplots(1)
            ax.set_xlabel(xLabel)
            ax.set_ylabel('$log_{10}(heating)$')
            ax.set_ylim([ymin1, ymax1])
            ax.grid()
            ax.set_title(title)
    
            for i in np.arange(nbRes):
                ax.plot(xAxis[i], yAxisStrobo[i], self.CS_1[i], label = labels[i])
                ax.plot(xAxis[i], yAxisSTFStrobo[i], self.CS_2[i])
            
            if(inset):
                ax2 = fig.add_axes(insetPos)
                ax2.set_ylim([ymin2, ymax2])
                ax2.set_title('all time', fontsize=10)
                ax2.tick_params(labelsize=8)
                for i in np.arange(nbRes):
                    ax2.plot(xAxis[i], yAxis[i],self.CS_1[i], label = labels[i])
                    ax2.plot(xAxis[i], yAxisSTF[i],self.CS_2[i])
    
            if (nbRes>1):
                ax.legend(loc='best')

    def PlotDynamics(self, xSort = 'omeff', numSimul = None, TF = None, diff = False,  population = 'g', inset = True, insetPos = None, nameGraph='',  *args):
        """
            population: g: ground internal state // e: excited internal state 
            // h: heating // 'n' pop in the n motional state // '+': e + g /sqrt(2) 
        """
        # PREPARE DATA
        assert ((len(args) % 2) == 0), 'plotCompareFidelity: pb in the args provided'
        nbExtraRes = int(len(args)/2)
        nbRes = 1 + nbExtraRes
            
        if (insetPos is None):
            insetPos = [0.49, 0.21, 0.4, 0.30]
            
        if (xSort != 'omeff'):
            assert False, 'plotDynamics: xSort not implemented yet'
            # SIMPLIFICATION ASSUME THAT weff are the same + sorted                
        else:
            xLabel = '$Time (T_eff)$'
        
        if (TF is None):
            listTF = ['1TF', '1TFmid', '1TFlast']
        else:
            listTF = [TF]        
        nbFrames = len(listTF)
        print(nbFrames)

            
        
        # iteration on w
        coeff = self.listOmEffScaled[numSimul]
        title = '$\omega = '+ str(coeff) + '\omega_0$'
        #nameRes = [nameGraph]
        #listSimsTmp = [self.listSimuls[i]]
        
        
        #GRAPH
        # Main heat at strobo times (straight) LTF (dotted) STF
        fig, axarray = plt.subplots(nbFrames)
        if(nbFrames == 1):
            axarray = [axarray]
        
        for tf in np.arange(nbFrames):
            t0, p0, _, _ = getPopulation1sim1TF(self, numSimul, population, listTF[tf], channel = 'TGT')
            t1, p1, t1s, p1s = getPopulation1sim1TF(self, numSimul, population, listTF[tf])
            axTmp = axarray[tf]
            axTmp.set_xlabel(xLabel)
            axTmp.set_ylabel('$P_' + population +'$')
            
            if(diff):
                diff = p1-p0
                axTmp.plot(t0, diff, self.CS_0[1], label = nameGraph)
            else:            
                axTmp.plot(t0, p0, self.CS_0[0], label = 'target')
                axTmp.plot(t1, p1, self.CS_3[1], label = nameGraph)           
                if(population in ['g', 'e', '+', '-']):
                    axTmp.set_ylim([0, 1])
            xmin = t1[0]
            xmax = t1[-1]
            
            for ires in np.arange(nbExtraRes):
                resTmp = args[ires*2]
                ttmp, ptmp, ptmps = getPopulation1sim1TF(resTmp, numSimul, population, listTF[tf])
                if(diff):
                    t0, p0, _, _ = getPopulation1sim1TF(resTmp, numSimul, population, listTF[tf], channel = 'TGT')
                    diff = ptmp-p0
                    axTmp.plot(ttmp, diff, self.CS_3[ires + 2], label = args[ires*2 + 1])
                else:
                    axTmp.plot(ttmp, ptmp, self.CS_3[ires + 2], label = args[ires*2 + 1])
            
            
            axTmp.set_xlim([xmin, xmax])
            axTmp.grid()
            if (nbRes>1):
                axTmp.legend(loc='best')
          
        plt.suptitle(title)
                
                
    
            

    
    # ========================== #
    # Get functions: relevant measures (get = compute)
    #     getAvgFidelity
    #     getFreqEff
    #     getTimeFrames
    #     getHeating
    # ========================== #
    def getAvgFidelity(self, oneSimul, tslice = None, fidChannel = 'FME_K_FNS'):
            """
                Compute fidelity for one simul one timeslice and one channel 
                (i.e. fidelities can be computed between different ref by def Numerical Simulations vs. FME with K)
            """
            fidT = oneSimul.fidT[fidChannel]
            if (slice is None):
                avgfid = np.average(fidT)
            else:
                avgfid = np.average(fidT[tslice])
            return avgfid

    def getFreqEff(self, onesimul):
        """ 
        Patch // should be now generated in simulation module

        """
        omeff = 0
        if(hasattr(onesimul, 'omeff')):
            omeff = onesimul.omeff
        else:
            if(onesimul.cType == 'strong'):
                modelTmp = onesimul.model 
                _, _, _, theta4, _, theta6, _ = toy.genCoeffsHStrong0(modelTmp.alphaFun, modelTmp.period, modelTmp.om, modelTmp.n0, modelTmp.nTr, modelTmp.eta)
                omeff = np.max(np.abs([theta6, theta4]))
            else:
                omeff = onesimul.cTarg
        onesimul.omeff = omeff
        return omeff

    def getTimeFrames(self, onesimul, laterTime = 0, lengthTF = 5, strobo = False):
        """
        Generate custom slice representing timeFrames
        2 timescales cohabits T (relating to the frequency of the drive) AND Teff (effective dynamics)
        laterTime and lengthTimeFrame are expressed in number of TEff 
        strobo refer to T timescale
        """
        dt = onesimul.p.dt
        nbppp = onesimul.p.n_pointsPerPeriod
        nT = nbppp
        #ndt = onesimul.p.n_dt
        #nperiods = onesimul.p.n_period
        #T = onesimul.p.T
        #t_max = onesimul.p.tmax
        Teff = np.pi / onesimul.omeff
        
        i_start = (laterTime * Teff) / dt
        i_stop = ((laterTime + lengthTF) * Teff) / dt
        if (strobo):
            index_start = int((i_start // nT) *nT)
            index_stop = int((i_stop // nT + 1) * nT)
            increment = nT
        else:

            index_start = int(np.floor(i_start))
            index_stop = int(np.ceil(i_stop))
            increment = 1

        TF = slice(index_start, index_stop, increment)
        return TF

    def getHeating(self, oneSimul, tslice = None, fidChannel = 'FNS'):
        """
            Compute heating for one simul one timeslice and one channel 
            (by deault heating is computed for the Floquet Numerical Simuls)
        """
        heatT = oneSimul.heatT[fidChannel]
        if (slice is None):
            maxHeat = np.max(heatT)
        else:
            maxHeat = np.max(heatT[tslice])
        return maxHeat



def getPopulation1sim1TF(res, numSimul = 0, population = 'g', TF = '1TF', channel = 'FNS'):
    """
        for one simulation, one type of population and one Time Frame
        return population and time associated
    
    """
    simTmp = res.listSimuls[numSimul]
    sl = res.TF[TF][numSimul]
    slstrobo = res.TFstro[TF][numSimul]
    tAxis = simTmp.model.tArray[sl] / res.listTEff[numSimul]
    tAxisstrobo = simTmp.model.tArray[slstrobo] / res.listTEff[numSimul]
    pop = sim.getPopulation(simTmp, population = population, channel = channel)
    popAxis = pop[sl]
    popAxisStrobo = pop[slstrobo]
                
    return [tAxis, popAxis, tAxisstrobo, popAxisStrobo]


