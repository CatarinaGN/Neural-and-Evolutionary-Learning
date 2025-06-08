def nested_cv_gp_slim_gsgp(X, y, param_grid, k_outer=5, k_inner=3, dataset_name="Chicken", gp_class=None, seed=42, alpha_sig=0.05):

    # === Imports === #
    import os
    import gc
    import torch
    import numpy as np
    import pandas as pd
    from IPython.display import display
    from itertools import product
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import KFold
    from scipy.stats import friedmanchisquare
    from sklearn.metrics import mean_squared_error
    import scikit_posthocs as sp
    from tqdm import tqdm  
    from datetime import datetime
    import plotly.graph_objects as go
    from slim.main_gp import gp
    from slim_gsgp.main_gsgp import gsgp
    from slim_gsgp.main_slim import slim

    # ========== Safe Inverse Transform ========== #
    def safe_inverse_transform(scaler, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return scaler.inverse_transform(y).flatten()

    # ===================== Nested CV Generator =====================
    def nested_cv_generator(X, y, k_outer=10, k_inner=3, random_state=None):
        """ https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/"""
        outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=random_state)
        for outer_train_idx, outer_test_idx in outer_cv.split(X):
            X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
            y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)
            X_test = X.iloc[outer_test_idx].reset_index(drop=True)
            y_test = y.iloc[outer_test_idx].reset_index(drop=True)

            inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=random_state)
            inner_splits = list(inner_cv.split(X_outer_train))

            yield {
                'X_outer_train': X_outer_train,
                'y_outer_train': y_outer_train,
                'X_test': X_test,
                'y_test': y_test,
                'inner_splits': inner_splits}

    # === Initial Setup === #
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    outer_scores = []
    best_grid_models = []
    validation_results = []
    detailed_results = []
    best_scalers = {}  

    # === Logging Setup === #
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
    #os.makedirs(f"./log/{gp_class}/{timestamp}/", exist_ok=True)

    # === Parameter Generation === #
    keys, values = zip(*param_grid.items())
    param_combos = [dict(zip(keys, v)) for v in product(*values)]

    # === Outer CV Loop === #
    for fold_idx, fold_data in enumerate(nested_cv_generator(X, y, k_outer=k_outer, k_inner=k_inner, random_state=seed)):
        print(f"\n========== OUTER FOLD {fold_idx+1}/{k_outer} ==========")
        
        best_config = None
        best_val_score = float('inf')
        all_inner_fold_scores = []
        current_fold_scalers = None

        # === Inner Grid Search === #
        for config_id, flat_config in enumerate(tqdm(param_combos, desc="Grid Search")):
            try: 
                gp_config = {
                    'initializer': flat_config['initializer'],
                    'sspace': {
                        'p_constants': flat_config['sspace.p_constants'],
                        'max_init_depth': flat_config['sspace.max_init_depth'],
                        'tree_constants': flat_config['sspace.tree_constants'],
                    },
                    'pop_size': flat_config['pop_size'],
                    'generations': flat_config['generations'],
                    'seed': seed
                }

                # Class-specific additions
                if 'SLIM' in str(gp_class.__name__):
                    gp_config.update({
                        'max_depth': flat_config['sspace.max_init_depth'] + 6,
                        'ms_lower': flat_config['ms_lower'],
                        'ms_upper': flat_config['ms_upper'],
                        'reconstruct': flat_config['reconstruct'],
                        'slim_version': flat_config['slim_version'],
                        'p_inflate': flat_config['p_inflate'],
                        'copy_parent': flat_config['copy_parent']
                    })
                elif 'gsgp' in str(gp_class.__name__):
                    gp_config.update({
                        'ms_lower': flat_config['ms_lower'],
                        'ms_upper': flat_config['ms_upper'],
                        'reconstruct': flat_config['reconstruct'],
                        'xo_prob': flat_config.get('xo_prob', 0.5)  # Default if not specified
                    })

                    print('gsgp')
                else:  # Standard GP
                    gp_config.update({
                        'max_depth': flat_config['sspace.max_depth'],
                        'xo_prob': flat_config.get('xo_prob', 0.5)  # Default if not specified
                    })

                inner_fold_rmses = []

                # === Inner CV Loop === #
                for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(fold_data['inner_splits']):
                    try:  
                        # === Data Scaling === #
                        x_scaler = RobustScaler()
                        y_scaler = RobustScaler()
                        X_train_scaled = x_scaler.fit_transform(fold_data['X_outer_train'].iloc[inner_train_idx])
                        X_val_scaled = x_scaler.transform(fold_data['X_outer_train'].iloc[inner_val_idx])
                        y_train_scaled = y_scaler.fit_transform(fold_data['y_outer_train'].iloc[inner_train_idx].values.reshape(-1, 1)).flatten()
                        y_val_scaled = y_scaler.transform(fold_data['y_outer_train'].iloc[inner_val_idx].values.reshape(-1, 1)).flatten()

                        # === Tensor Conversion === #
                        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
                        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
                        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

                        # === Model Training === #
                        model_params = {
                            # Problem Instance
                            'X_train':X_train_tensor, 'y_train':y_train_tensor,
                            'X_test':X_val_tensor, 'y_test':y_val_tensor,
                            'dataset_name':dataset_name,
                            'fitness_function':'rmse',
                            'minimization':True,

                            # Search space
                            'init_depth':gp_config['sspace']['max_init_depth'],
                            'tree_constants': gp_config['sspace']['tree_constants'],
                            'tree_functions': ['add', 'subtract', 'multiply', 'divide'],
                            'prob_const': gp_config['sspace']['p_constants'],
                            'initializer': gp_config['initializer'],

                            # SLIM instance
                            'pop_size': gp_config['pop_size'],
                            'tournament_size': 2,

                            # Solve settings
                            'n_iter': gp_config['generations'],
                            'elitism': True,
                            'n_elites': 1,
                            'test_elite': True,
                            'log_level': 0,
                            'verbose': 0,
                            'n_jobs': 1,
                            'seed': gp_config['seed']}

                        # === Class-Specific Parameters === #
                        if 'SLIM' in str(gp_class.__name__):
                            model_params.update({
                                'max_depth': gp_config['max_depth'], 
                                'ms_lower': gp_config['ms_lower'],
                                'ms_upper': gp_config['ms_upper'],
                                'reconstruct': gp_config['reconstruct'],
                                'slim_version': gp_config['slim_version'],
                                'p_inflate': gp_config['p_inflate'],
                                'copy_parent': gp_config['copy_parent']
                            })
                        elif 'gsgp' in str(gp_class.__name__):
                            model_params.update({
                                'ms_lower': gp_config['ms_lower'],
                                'ms_upper': gp_config['ms_upper'],
                                'reconstruct': gp_config['reconstruct'],
                                'p_xo': gp_config.get('xo_prob', 0.5)  # Default if not in config
                            })
                        else:  # Standard GP
                            model_params.update({
                                'max_depth': gp_config['max_depth'],
                                'p_xo': gp_config.get('xo_prob', 0.5)  # Default if not in config
                            })

                        # Instantiate the model
                        model = gp_class(**model_params)

                        # === Evaluation === #
                        with torch.no_grad():
                            y_pred_scaled = model.predict(X_val_tensor).numpy()

                            y_true_scaled = y_val_tensor.numpy()

                            y_pred = safe_inverse_transform(y_scaler, y_pred_scaled)
                            y_true = safe_inverse_transform(y_scaler, y_true_scaled)

                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                        inner_fold_rmses.append(rmse)
                        detailed_results.append({
                            'outer_fold': fold_idx + 1,
                            'inner_fold': inner_fold_idx + 1,
                            'config_id': config_id,
                            'config': str(flat_config),
                            'rmse': rmse
                        })

                    finally:
                        del model 
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # === Update Best Config === #
                avg_val_rmse = np.mean(inner_fold_rmses)
                all_inner_fold_scores.append(inner_fold_rmses)
                print(f"  Config {config_id+1}/{len(param_combos)} | Inner Mean RMSE: {avg_val_rmse:.4f}")

                if avg_val_rmse < best_val_score:
                    best_val_score = avg_val_rmse
                    best_config = gp_config

            except Exception as e:
                print(f"Error in config {config_id}: {str(e)}")
                continue

        print(f"\n✅ Best config for outer fold {fold_idx+1}:")
        print(best_config)
        print(f"Best inner RMSE: {best_val_score:.4f}")
        best_grid_models.append(best_config)

        # === Statistical Tests === #
        df_inner_scores = pd.DataFrame(all_inner_fold_scores).T
        validation_results.append(df_inner_scores)

        # ============ Friedman & Posthoc Tests ============ #
        print("\nStatistical test on inner fold scores:")
        stat, p = friedmanchisquare(*[df_inner_scores[col] for col in df_inner_scores.columns])
        print(f"Friedman test statistic: {stat:.4f}, p-value: {p:.4f}")
        if p < alpha_sig:
            posthoc_result = sp.posthoc_nemenyi_friedman(df_inner_scores.to_numpy())
            #print("\nFull posthoc Nemenyi test p-values:")
            #print(posthoc_result.round(4))

            significant_mask = posthoc_result < alpha_sig
            if significant_mask.values.any():
                 print("\nSignificant pairwise differences (p < alpha):")
                 sig_table = posthoc_result.where(significant_mask).dropna(how='all').dropna(axis=1, how='all')
                 display(sig_table.round(4))
            else:
                 print("Friedman test was significant, but no significant pairwise differences found.")
        else:
            print("No significant differences found between configs.")

        # ============ Boxplot ============ #
        df_fold_results = pd.DataFrame([r for r in detailed_results if r['outer_fold'] == fold_idx + 1])
        config_labels = {
            i: f"Config {i+1}<br>" + "<br>".join([f"{k}: {v}"[:30] for k, v in param_combos[i].items()])
            for i in range(len(param_combos))
        }
        df_fold_results['config_label'] = df_fold_results['config_id'].map(config_labels)

        fig = go.Figure()
        fig.add_trace(go.Box(
            x=df_fold_results['config_label'],
            y=df_fold_results['rmse'],
            fillcolor='rgba(108, 140, 200, 0.3)',
            line=dict(color='rgba(108, 140, 200, 1)'),
            boxpoints='all',
            jitter=0,
            pointpos=0,
            marker=dict(color='rgba(108, 140, 200, 1)')
        ))
        fig.update_layout(
            title=f'Inner CV RMSEs - Outer Fold {fold_idx+1}',
            yaxis_title='Validation RMSE',
            width=300 * len(param_combos),
            height=400,
            plot_bgcolor='#f1f1f1',
            xaxis_tickangle=-90,
            margin=dict(l=50, r=50, t=50, b=20),
            showlegend=False
        )
        fig.show()

        # === Outer Evaluation === #
        try:
            x_scaler = RobustScaler()
            y_scaler = RobustScaler()
            X_outer_train_scaled = x_scaler.fit_transform(fold_data['X_outer_train'])
            X_test_scaled = x_scaler.transform(fold_data['X_test'])
            y_outer_train_scaled = y_scaler.fit_transform(fold_data['y_outer_train'].values.reshape(-1, 1)).flatten()
            y_test_scaled = y_scaler.transform(fold_data['y_test'].values.reshape(-1, 1)).flatten()

            X_outer_train_tensor = torch.tensor(X_outer_train_scaled, dtype=torch.float32)
            y_outer_train_tensor = torch.tensor(y_outer_train_scaled, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

            model_params = {
                # Problem Instance
                'X_train':X_outer_train_tensor, 'y_train':y_outer_train_tensor,
                'X_test':X_test_tensor, 'y_test':y_test_tensor,
                'dataset_name':dataset_name,
                'fitness_function':'rmse',
                'minimization':True,

                # Search space
                'init_depth':best_config['sspace']['max_init_depth'],
                'tree_constants': best_config['sspace']['tree_constants'],
                'tree_functions': ['add', 'subtract', 'multiply', 'divide'],
                'prob_const': best_config['sspace']['p_constants'],
                'initializer': best_config['initializer'],

                # SLIM instance
                'pop_size': best_config['pop_size'],
                'tournament_size': 2,

                # Solve settings
                'n_iter': best_config['generations'],
                'elitism': True,
                'n_elites': 1,
                'test_elite': True,
                'log_level': 0,
                'verbose': 0,
                'n_jobs': 1,
                'seed': best_config['seed']
                
            }

            # === Class-Specific Parameters === #
            if 'SLIM' in str(gp_class.__name__):
                model_params.update({
                    'max_depth': best_config['max_depth'], 
                    'ms_lower': best_config['ms_lower'],
                    'ms_upper': best_config['ms_upper'],
                    'reconstruct': best_config['reconstruct'],
                    'slim_version': best_config['slim_version'],
                    'p_inflate': best_config['p_inflate'],
                    'copy_parent': best_config['copy_parent']
                })
            elif 'gsgp' in str(gp_class.__name__):
                model_params.update({
                    'tournament_size': 2,
                    'ms_lower': best_config['ms_lower'],
                    'ms_upper': best_config['ms_upper'],
                    'reconstruct': best_config['reconstruct'],
                    'p_xo': best_config.get('xo_prob', 0.5)  # Default if not in config
                })
            else:  # Standard GP
                model_params.update({
                    'max_depth': best_config['max_depth'],
                    'tournament_size': 2,
                    'p_xo': best_config.get('xo_prob', 0.5)  # Default if not in config
                })

            # Instantiate the model
            model = gp_class(**model_params)


            with torch.no_grad():
                y_pred_scaled = model.predict(X_test_tensor).numpy()
                y_pred_rescaled = safe_inverse_transform(y_scaler, y_pred_scaled)
                test_rmse = np.sqrt(mean_squared_error(fold_data['y_test'], y_pred_rescaled))

            print(f"\n[Outer Fold {fold_idx+1}] Test RMSE: {test_rmse:.4f}")
            outer_scores.append(test_rmse)

        finally:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.array(outer_scores), best_grid_models, pd.DataFrame(detailed_results), validation_results