import matlab.engine


if __name__ == '__main__':
    repo = "D:\\PhD\\Data\\test\\2021\\output_r2n\\"
    fname = "z5700001________z_57_00001______dcu1____________dcuconddata______210215_154202.bin"
    code_path = "D:\\PhD\\Code\\health_detection\\clustering-v2\\matlab"
    eng = matlab.engine.start_matlab()
    eng.addpath(code_path)
    error_nb = eng.preprocess_cycle(repo + fname, nargout=1)
    eng.quit()
