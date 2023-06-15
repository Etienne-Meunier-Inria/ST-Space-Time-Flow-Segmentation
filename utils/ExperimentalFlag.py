class ExperimentalFlag :
    __experiment_flag = None

    @staticmethod
    def get():
        return ExperimentalFlag.__experiment_flag

    @staticmethod
    def check(flag) :
        if ExperimentalFlag.__experiment_flag is None :
            return False
        return (flag in ExperimentalFlag.__experiment_flag)

    @staticmethod
    def set(flag) :
        ExperimentalFlag.__experiment_flag = flag
        if flag is not None :
            print(f'Setting ExperimentalFlag : {flag}')
