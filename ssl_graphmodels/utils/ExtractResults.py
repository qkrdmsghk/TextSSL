from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os

class ExtractResults( ):
    def __init__(self, model, device, dictionary=None):
        self.model = model
        self.device = device
        self.model.eval()
        self.cooc_path = '/data/project/yinhuapark/DATA_PRE/in-hospital-mortality/top_20_all/test_all_cooc'
        self.seq_path = '/data/project/yinhuapark/DATA_PRE/in-hospital-mortality/top_20_all/test_all_seq'
        self.dictionary = dictionary

    def test2evaluate(self, loader):
        y_pred = []
        y_true = []

        for data in tqdm(loader, desc='Evaluating on test set'):
            data = data.to(self.device)
            with torch.no_grad():
                pt_out_ = self.model(data)

            true = data.y_p
            pt_out = pt_out_.cpu().detach().numpy()
            # print(pt_out)
            true = true.cpu().detach().numpy()
            # torch.cuda.empty_cache()
            y_pred.append(pt_out[:, 1])
            y_true.append(true)
            del data, pt_out_, true,pt_out

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        df = pd.DataFrame(np.array([y_pred, y_true]).T, columns=['prediction', 'true'])
        del y_true, y_pred
        return df


    def dive2pool(self, loader):
        y_pred = []
        y_true = []
        intersection_ratio = []
        for data in tqdm(loader, desc='Extracting pooled nodes on test set'):
            data = data.to(self.device)
            with torch.no_grad():
                x_n = data.x_n_id
                intersection = torch.unique(x_n)[[torch.unique(x_n, return_counts=True)[1] > 1]]
                pt_out, _, pooled_id = self.model(data, pool_info=True)
                if intersection.shape[0] == 0:
                    pooled_intersection_ratio = -1
                else:
                    pooled_intersection_ratio = (np.intersect1d(pooled_id.cpu(), intersection.cpu())).shape[0] / intersection.shape[0]
                true = data.y_p
            pt_out = pt_out.cpu().detach().numpy()
            true = true.cpu().detach().numpy()
            torch.cuda.empty_cache()
            y_pred.append(pt_out[:, 1])
            y_true.append(true)
            intersection_ratio.append([pooled_intersection_ratio, pooled_id.shape[0]])
            del data, true,pt_out
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        intersection_ratio = np.array(intersection_ratio)
        df = pd.DataFrame(np.array([y_pred, y_true]).T, columns=['prediction', 'true'])
        df['ratio'] = intersection_ratio[:, 0]
        df['pool_num'] = intersection_ratio[:, 1]
        return df

    def cs_generate_dic(self, n_id, batch_n, intersection):
        note_id = batch_n[0].item()
        note_dics = []
        note_dic = {}
        for node_i, note_i in zip(n_id.squeeze(), batch_n.squeeze()):
            note_dic[self.dictionary[node_i]] = intersection[1][intersection[0] == node_i].item()
            if note_id != note_i:
                note_dics.append(note_dic)
                note_dic = {}
                note_id = note_i
        note_dics.append(note_dic)
        return note_dics

    def cs_generate_str(self, patient_seq):
        filtered_text = patient_seq['filtered_text']
        default_dic = patient_seq['default_dic']
        pool_dic = patient_seq['pool_dic']

        df_strs = []
        pool_strs = []
        for note, ddic, pdic in zip(filtered_text, default_dic, pool_dic):
            print(note)
            print(ddic)
            print(pdic)
            strr = '\n'
            pool_strr = '\n'
            for words in eval(note):
                for word in words:
                    if word in ddic:
                        if ddic[word] == 1:
                            strr += word + ' '
                        else:
                            strr += '(' + word + ')' + ' '
                    else:
                        print('not in ddic')
                    if word in pdic:
                        if pdic[word] == 1:
                            pool_strr += word + ' '
                        else:
                            pool_strr += '(' + word + ')' + ' '
                    else:
                        print('not in pdic')
                strr += '\n'
                pool_strr += '\n'
            df_strs.append(strr)
            pool_strs.append(pool_strr)

        patient_seq['default_str'] = df_strs
        patient_seq['pool_str'] = pool_strs
        patient_seq['filtered_text'] = patient_seq['filtered_text'].apply(lambda x: '\n'+x+'\n')
        patient_seq['default_dic'] = patient_seq['default_dic'].apply(lambda x: '\n'+str(x)+'\n')
        patient_seq['pool_dic'] = patient_seq['pool_dic'].apply(lambda x: '\n'+str(x)+'\n')


        return patient_seq

    def case_study(self, loader, pid):
        for i, data in  enumerate(tqdm(loader, desc='Extracting pooled nodes on test set')):
            data = data.to(self.device)
            intersection = torch.unique(data.x_n_id, return_counts=True)
            patients = list(filter(lambda x: x.find("episode") != -1, os.listdir(self.cooc_path)))
            if pid != None:
                patient_seq = pd.read_csv(os.path.join(self.seq_path, patients[pid]), sep='\t', header=0)
            else:
                if data.y_p.cpu().detach().numpy() == [1]:
                    pt_out, pooled_batch_n, pooled_n_id = self.model(data, pool_info=True)
                    if pt_out.cpu().detach().numpy()[0, 1].round() ==1:
                        patient_seq = pd.read_csv(os.path.join(self.seq_path, patients[i]), sep='\t', header=0)
                        pid = i
                        break

        note_dics = self.cs_generate_dic(data.x_n_id, data.batch_n, intersection)
        patient_seq['default_dic'] = note_dics


        note_dics = self.cs_generate_dic(pooled_n_id.squeeze(), pooled_batch_n.squeeze(), intersection)
        print(note_dics)

        patient_seq['pool_dic'] = note_dics
        patient_seq = self.cs_generate_str(patient_seq)
        return patient_seq, patients[pid]
