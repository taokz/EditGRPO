# Security / privacy notes on the original repository

Locations only; values are intentionally not reproduced here.

1. `examples/data/sample_3000_mimic_cxr.json`
   - Image paths carry a personal path prefix (`/mnt/ml_data/<user>/...`).
   - Contains MIMIC-CXR subject/study identifiers (`id`, image paths) and report text. MIMIC-CXR is governed by the PhysioNet credentialed-data use agreement: redistributing derived reports/ids in a public repo is a DUA concern. Recommend removing the file from the public history and shipping a synthetic sample (nano has `SyntheticCXRDataset`).
2. Cluster-specific settings in sbatch files:
   - `examples/mimic_cxr_editgrpo_ops1_para_half_tau_0_6.sbatch` (partition `prod_long` at :11, `/mnt/ckpt/...` at :31 and :112, conda env `editgrpo` at :24).
   - `flask_server/run_server_mimic_cxr.sbatch` (partitions `prod_long`/`legacy` at :12-13, constraint `GPUMEM_48GB` at :14, conda env `metric`).
   - `flask_server/run_RaTEscore_extractor.sbatch` (partition `legacy`, env `metric`), `flask_server/run_green_server.sbatch` (env `green`).
3. Flask servers bind `0.0.0.0` with no authentication or TLS: `flask_server/all_flask_server.py:262`, `RaTEscore_extractor.py:479`, `radgraph_extractor.py:100`, `green_flask_server.py:205`. Anyone on the network can submit text to be scored (and the endpoints run model inference). Bind to localhost or add a token.
4. `verl/trainer/main.py:122`: commented-out `_redis_password="your_password_here"` placeholder; harmless, but remove to avoid encouraging inline secrets.
5. Server host placeholders (`REWARD_SERVER_HOST`) are fine.

nano-EditGRPO copies none of the above: no data files, paths, cluster settings, hosts or credentials; rewards run in-process and test data is synthetic.
