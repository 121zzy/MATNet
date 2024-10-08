
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'CDD':
            self.label_transform = "norm"
            self.root_dir = '/media/dsk2/zzy/project/CDD_ch/'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = '/media/dsk2/zzy/project/DSIFN_ch/'
        elif data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/media/dsk2/zzy/project/LEVIR_ch/'
        elif data_name == 'SYSU':
            self.label_transform = "norm"
            self.root_dir = '/media/dsk2/zzy/project/SYSU-ch/'
        elif data_name == 'TYPO':
            self.label_transform = "norm"
            self.root_dir = '/media/lidan/ssd2/CDData/TYPO/'
        elif data_name == 'quick_start_LEVIR':
            self.root_dir = './samples_LEVIR/'
        elif data_name == 'quick_start_DSIFN':
            self.root_dir = './samples_DSIFN/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='CDD')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

