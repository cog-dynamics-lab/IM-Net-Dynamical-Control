import importlib.util, os, time
os.chdir("/mnt/proj/V2027/000")
out_dir = "/mnt/proj/V2027/000/results_reproduction/negk1_n20"
os.environ["REPRO_DIR"] = out_dir

spec = importlib.util.spec_from_file_location("cco", "experiments/01_cco_simulation.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 强制覆盖模块内可能写死的保存路径变量
for attr in ["REPRO_DIR", "OUT_DIR", "RESULTS_DIR", "SAVE_DIR"]:
    if hasattr(m, attr):
        setattr(m, attr, out_dir)

t0 = time.time()
k = m.make_fig6(n_init=20)
print(f"[done n_init=20] kappa_c={k}, took {(time.time()-t0)/60:.1f}min", flush=True)
