
import numpy as np
from math import sqrt
import io

def predictionToLabel(result,nclass):
    result_predict = np.zeros(len(result),dtype=int)
    class_labels = np.arange(nclass)
    for i in range(0,len(result)):
        pred = result[i]
        result_predict[i] = class_labels[np.argmax(pred)]
    
    return result_predict
    
def saveCM(cm,savePathName):
    # Save confusion matrix result
    savedata = cm
    outfn = savePathName
    print('====> Save CM ', outfn)
    np.save(outfn, savedata)


def cm_code_paper(model_root_dir,cm_final,ckpttype='',custom=0):
    #custom ga di pake di sni karena sudah di padding di luar
    if custom == 0:
        Cs5 = np.zeros((5,5))
        Cs5[1:,1:] = cm_final[-4:,-4:]
        Cs5[0,1:] = np.sum(cm_final[:3,3:], 0)
        Cs5[1:,0] = np.sum(cm_final[3:,:3], 1)
        Cs5[0,0] = np.sum(cm_final[:3,:3])
    else:
        Cs5 = np.full((5,5),custom)
        Cs5[1:,1:] += cm_final[-4:,-4:]
        Cs5[0,1:] += np.sum(cm_final[:3,3:], 0)
        Cs5[1:,0] += np.sum(cm_final[3:,:3], 1)
        Cs5[0,0] += np.sum(cm_final[:3,:3])

    print(Cs5.astype(int))
    saveCM(Cs5,model_root_dir+'/cm5_result_'+ckpttype+'.npy')
    #Cs5 = cm_final_step2

    selected_class = 2
    TpV = Cs5[selected_class,selected_class]
    FpV = np.sum(Cs5[:,selected_class]) - TpV
    FnV = np.sum(Cs5[selected_class,:]) - TpV
    TnV = np.sum(Cs5[:,:]) - FpV - FnV - TpV
    accV = np.round(100*(TpV + TnV)/(TpV+TnV+FpV+FnV),2)
    senV = np.round(100*(TpV)/(TpV+FnV),2)
    speV = np.round(100*(TnV)/(TnV+FpV),2)
    pprV = np.round(100*(TpV)/(TpV+FpV),2)
    F1V = np.round(2/((1/senV)+(1/pprV)),2)
    GsV = np.round(sqrt((senV*pprV)),2)

    selected_class = 1
    TpS = Cs5[selected_class,selected_class]
    FpS = np.sum(Cs5[:,selected_class]) - TpS
    FnS = np.sum(Cs5[selected_class,:]) - TpS
    TnS = np.sum(Cs5[:,:]) - FpS - FnS - TpS
    accS = np.round(100*(TpS + TnS)/(TpS+TnS+FpS+FnS),2)
    senS = np.round(100*(TpS)/(TpS+FnS),2)
    speS = np.round(100*(TnS)/(TnS+FpS),2)
    pprS = np.round(100*(TpS)/(TpS+FpS),2)
    F1S = np.round(2/((1/senS)+(1/pprS)),2)
    GsS = np.round(sqrt((senS*pprS)),2)

    # outputMat = np.reshape(np.asarray([accV, senV, speV, pprV, accS, senS, speS, pprS]), (1,-1))

    # print(outputMat)

    vprint = "VEB\nAcc {} \nSen {} \nSpe {} \nPpr {} \nF1 {} \nGscore {}".format(accV,senV, speV, pprV, F1V, GsV)
    print(vprint)

    sprint = "\nSVEB\nAcc {} \nSen {} \nSpe {} \nPpr {} \nF1 {} \nGscore {}".format(accS,senS, speS, pprS, F1S, GsS)
    print(sprint)
    

    """
    ## note
    pnote = "\n"
    pnote = pnote+model_root_dir+"\n"
    pnote = pnote+vprint+sprint
    pathNote = model_root_dir+"/result_note_ori_code_"+customMsg+".txt"
    with open(pathNote,'a') as f:
            f.write(pnote)
    """
    csvFormat = "\r"  
    csvFormat += ckpttype+",Acc,Sen, Spe, Ppr, F1, Gscore"
    csvFormat += "\rVEB,{},{},{},{},{},{}".format(accV,senV, speV, pprV, F1V, GsV)
    csvFormat += "\rSVEB,{},{},{},{},{},{}".format(accS,senS, speS, pprS, F1S, GsS)
    total = F1V+F1S+GsV+GsS # cara simple hitung point
    total2 = accV+senV+speV+pprV+accS+senS+speS+pprS
    resultDict = {"ckptType":ckpttype,
                    "F1_veb":F1V,"F1_sveb":F1S,
                    "Gs_veb":GsV,"Gs_sveb":GsS,
                    "accV":accV,"senV":senV,"speV":speV,"pprV":pprV,
                    "accS":accS,"senS":senS,"speS":speS,"pprS":pprS,
                    "total":total,"total2":total2} 
    
    print("total point :{}".format(total))
    return csvFormat,resultDict
    
def saveResult(model_root_dir,csvFormat,customMsg):
    pathNote = model_root_dir+"\\result_note_ori_code_"+customMsg+".csv"
    with open(pathNote,'a') as f:
        f.write(csvFormat)
        
        
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def save_modelSummary(summary_string,model_summary_path): # for saving model.summary()
    # currentPath+'/modelSummary/'+model_summary_file_name+'.txt'
    with open(model_summary_path,'w+') as f:
        print(summary_string, file=f)
            
def cm_code_self(model_root_dir,cm_final,customMsg=''):
    print(cm_final)
    print()
    temp_final = cm_final[:,0] + cm_final[:,1] + cm_final[:,2]
    temp_final = temp_final.reshape(1,temp_final.shape[0])
    print(temp_final.T)
    print()
    cm_final_step1 = np.concatenate((temp_final.T,cm_final[:,3:]),axis=1)
    print(cm_final_step1)
    print()
    temp_final = cm_final_step1[0,:]+cm_final_step1[1,:]+cm_final_step1[2,:]
    temp_final = temp_final.reshape(1,temp_final.shape[0])
    print(temp_final)
    print()
    cm_final_step2 = np.concatenate((temp_final,cm_final_step1[3:,:]),axis=0)
    print(cm_final_step2)

    print()
    print('====== SVEB =======')

    print()
    print(cm_final_step2)

    cm_temp_sveb = np.vstack((cm_final_step2[1,:],cm_final_step2[0,:],cm_final_step2[2,:],
                              cm_final_step2[3,:],cm_final_step2[4,:]))
    print()
    print(cm_temp_sveb)

    pos0 = cm_temp_sveb[:,1].reshape(1,cm_temp_sveb[:,1].shape[0])
    pos1 = cm_temp_sveb[:,0].reshape(1,cm_temp_sveb[:,0].shape[0])
    pos2 = cm_temp_sveb[:,2].reshape(1,cm_temp_sveb[:,2].shape[0])
    pos3 = cm_temp_sveb[:,3].reshape(1,cm_temp_sveb[:,3].shape[0])
    pos4 = cm_temp_sveb[:,4].reshape(1,cm_temp_sveb[:,4].shape[0])
    cm_temp_sveb = np.hstack((pos0.T,pos1.T,pos2.T,pos3.T,pos4.T))
    print()
    print(cm_temp_sveb)

    print()
    print('====== VEB =======')
    print()
    print(cm_final_step2)

    cm_temp_veb = np.vstack((cm_final_step2[2,:],cm_final_step2[1,:],cm_final_step2[0,:],
                              cm_final_step2[3,:],cm_final_step2[4,:]))
    print()
    print(cm_temp_veb)

    pos0 = cm_temp_veb[:,2].reshape(1,cm_temp_veb[:,2].shape[0])
    pos1 = cm_temp_veb[:,1].reshape(1,cm_temp_veb[:,1].shape[0])
    pos2 = cm_temp_veb[:,0].reshape(1,cm_temp_veb[:,0].shape[0])
    pos3 = cm_temp_veb[:,3].reshape(1,cm_temp_veb[:,3].shape[0])
    pos4 = cm_temp_veb[:,4].reshape(1,cm_temp_veb[:,4].shape[0])
    cm_temp_veb = np.hstack((pos0.T,pos1.T,pos2.T,pos3.T,pos4.T))
    print()
    print(cm_temp_veb)


    ## for SVEB
    cm = cm_temp_sveb
    TP = cm[0,0]
    TN = sum(sum(cm[1:,1:]))
    FP = sum(cm[1:,0])
    FN = sum(cm[0,1:])

    s1print = "\nSVEB\nTP {} \nTN {} \nFP {} \nFN {}".format(TP, TN, FP, FN)
    print(s1print)

    AccS = np.round(100*(TP+TN)/(TP+TN+FP+FN),2)
    SenS = np.round(100*TP/(TP+FN),2)
    SpeS = np.round(100*TN/(TN+FP),2)
    PprS = np.round(100*TP/(TP+FP),2)
    F1S = np.round(2/((1/SenS)+(1/PprS)),2)
    GsS = np.round(sqrt((SenS*PprS)),2)

    s2print = "\nAcc {} \nSen {} \nSpe {} \nPpr {} \nF1 {} \nGscore {}".format(AccS,SenS, SpeS, PprS, F1S, GsS)
    print(s2print)

    ## for VEB
    cm = cm_temp_veb
    TP = cm[0,0]
    TN = sum(sum(cm[1:,1:]))
    FP = sum(cm[1:,0])
    FN = sum(cm[0,1:])

    v1print = "\nVEB\nTP {} \nTN {} \nFP {} \nFN {}".format(TP, TN, FP, FN)
    print(v1print)

    AccV = np.round(100*(TP+TN)/(TP+TN+FP+FN),2)
    SenV = np.round(100*TP/(TP+FN),2)
    SpeV = np.round(100*TN/(TN+FP),2)
    PprV = np.round(100*TP/(TP+FP),2)
    F1V = np.round(2/((1/SenV)+(1/PprV)),2)
    GsV = np.round(sqrt((SenV*PprV)),2)

    v2print = "\nAcc {} \nSen {} \nSpe {} \nPpr {} \nF1 {} \nGscore {}".format(AccV,SenV, SpeV, PprV, F1V, GsV)
    print(v2print)


    ## note
    pnote = "\n"
    pnote = pnote+model_root_dir+"\n"
    pnote = pnote+s1print+s2print+v1print+v2print
    pathNote = model_root_dir+"/result_note_"+customMsg+".txt"
    with open(pathNote,'a') as f:
            f.write(pnote)
            
    csvFormat = "Class,Acc,Sen, Spe, Ppr, F1, Gscore"
    csvFormat += "\rVEB,{},{},{},{},{},{}".format(AccV,SenV, SpeV, PprV, F1V, GsV)
    csvFormat += "\rSVEB,{},{},{},{},{},{}".format(AccS,SenS, SpeS, PprS, F1S, GsS)
    pathNote = model_root_dir+"/result_note_selfmade_"+customMsg+".csv"
    with open(pathNote,'a') as f:
            f.write(csvFormat)