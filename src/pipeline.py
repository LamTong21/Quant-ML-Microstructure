# src/pipeline.py

import argparse
import yaml
import pandas as pd
import numpy as np

from data_loader import VnstockLoader, IntegrityAudit, ReturnTopology, TripleBarrierLabeling
from dgp_diagnostics import DGPScanner
from feature_factory import Layer10FeatureRouter
from feature_selection import SelectionPipelineRouter, KinematicLagTransform
from engine import PurgedTimeSeriesSplit, MLTrainer, SignalGenerator, VectorizedBacktestSimulator

def load_config(config_path: str) -> dict:
    """Nạp cấu hình từ file YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def align_features_and_target(X: pd.DataFrame, df_raw: pd.DataFrame) -> tuple:
    """Căn chỉnh ma trận đặc trưng và nhãn mục tiêu, loại bỏ NaN[cite: 2, 3]."""
    y = df_raw['target_label'].copy().rename('target')
    aligned = pd.concat([X, y], axis=1).dropna()
    return aligned.drop(columns=['target']), aligned['target']

def main(config_path: str):
    config = load_config(config_path)
    
    print("=" * 80)
    print(f"BẮT ĐẦU PIPELINE THỰC NGHIỆM: {config['symbol']}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: Data Integrity & Volatility-Scaled Dynamic Triple-Barrier Labeling
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Nạp & Kiểm toán dữ liệu[cite: 1, 2, 3]")
    loader = VnstockLoader(
        symbol=config['symbol'], 
        start_date=config['start_date'], 
        end_date=config['end_date'], 
        source=config.get('data_source', 'KBS')
    )
    df_raw = loader.fetch_data()
    
    df_clean, audit_stats = IntegrityAudit.run_audit(df_raw)
    print(f"  -> Trạng thái kiểm toán: {audit_stats}")
    
    df_topo = ReturnTopology.compute(df_clean, max_depth_levels=config.get('max_l2_depth', 10))
    df_final = TripleBarrierLabeling.label(
        df_topo, 
        h=config['triple_barrier']['h'], 
        pt=config['triple_barrier']['pt'], 
        sl=config['triple_barrier']['sl'], 
        vol_span=config['triple_barrier']['vol_span']
    )

    # -------------------------------------------------------------------------
    # KHỞI TẠO PURGED WALK-FORWARD CV
    # -------------------------------------------------------------------------
    n_splits = config['cv']['n_splits']
    purge_gap = config['cv']['purge_gap']
    buffer_days = config['cv']['buffer_days']
    
    ptscv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    
    oos_predictions = []
    trained_models = []

    print("\n" + "=" * 80)
    print("BẮT ĐẦU END-TO-END PURGED WALK-FORWARD VALIDATION[cite: 1, 2, 3]")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(ptscv.split(df_final)):
        print(f"\n[FOLD {fold + 1}/{n_splits}]")

        df_train_raw = df_final.iloc[train_idx].copy()
        df_test_raw = df_final.iloc[test_idx].copy()

        # Tạo vùng đệm (Warm-up Buffer) để tránh NaN khi tính Rolling Features cho Test Set[cite: 1, 2, 3]
        buffer_data = df_train_raw.iloc[-buffer_days:].copy()
        df_test_with_buffer = pd.concat([buffer_data, df_test_raw])

        # ---------------------------------------------------------------------
        # STAGE 2: DGP Diagnostics (Khảo sát quá trình sinh dữ liệu trên Train)
        # ---------------------------------------------------------------------
        print("  -> [STAGE 2] Chạy bộ Cảm biến DGP Diagnostics[cite: 1, 2]")
        routing_payload = DGPScanner.execute(df_train_raw)

        # Cập nhật ngưỡng tham số từ config vào payload
        routing_payload['max_clusters'] = config['selection']['max_clusters']
        routing_payload['mi_threshold'] = config['selection']['mi_threshold']

        # ---------------------------------------------------------------------
        # STAGE 3: Evidence-Based Feature Factory
        # ---------------------------------------------------------------------
        print("  -> [STAGE 3] Kích hoạt Institutional Feature Factory[cite: 1, 2]")
        feature_router = Layer10FeatureRouter(routing_payload)
        X_train_candidates = feature_router.execute(df_train_raw)
        
        # Căn chỉnh lại Train Set trước khi đưa vào Stage 4
        X_train_aligned, y_train_aligned = align_features_and_target(X_train_candidates, df_train_raw)

        # ---------------------------------------------------------------------
        # STAGE 4: Statistical Pruning & Kinematic Transformations
        # ---------------------------------------------------------------------
        print("  -> [STAGE 4] Lọc Thống kê đa tầng & Biến đổi Động học[cite: 1, 2, 3]")
        selection_pipeline = SelectionPipelineRouter(routing_payload)
        X_train_final, optimal_lags, valid_features = selection_pipeline.execute(X_train_aligned, y_train_aligned)

        # Xử lý Test Set thông qua Stage 3 & Stage 4
        X_test_candidates = feature_router.execute(df_test_with_buffer)
        X_test_transformed = KinematicLagTransform.apply_transform(X_test_candidates, optimal_lags)
        
        # Đồng bộ cột 'regime_p_high_vol' nếu có
        if 'regime_p_high_vol' in X_test_candidates.columns:
            X_test_transformed['regime_p_high_vol'] = X_test_candidates['regime_p_high_vol']
            
        # Lọc các biến đã qua xác thực tại Train và khôi phục index gốc của Test
        available_cols = [c for c in valid_features if c in X_test_transformed.columns]
        if 'regime_p_high_vol' in X_test_transformed.columns and 'regime_p_high_vol' not in available_cols:
            available_cols.append('regime_p_high_vol')
            
        X_test_buffer_final = X_test_transformed[available_cols]
        X_test_features_raw = X_test_buffer_final.loc[X_test_buffer_final.index >= df_test_raw.index[0]]
        
        _, y_test_buffer = align_features_and_target(X_test_candidates, df_test_with_buffer)
        common_index = X_test_features_raw.index.intersection(y_test_buffer.index)
        
        X_test_final = X_test_features_raw.loc[common_index]
        y_test_final = y_test_buffer.loc[common_index]

        if X_test_final.empty:
            print("  -> [Cảnh báo] Test set rỗng sau khi căn chỉnh. Bỏ qua fold.")
            continue

        print(f"  -> Kích thước Train: {X_train_final.shape} | Test: {X_test_final.shape}")

        # ---------------------------------------------------------------------
        # STAGE 5: Purged Walk-Forward ML Engine & Realistic Execution
        # ---------------------------------------------------------------------
        print("  -> [STAGE 5] Tuning Hyper-parameters & Huấn luyện XGBoost[cite: 1, 2, 3]")
        trainer = MLTrainer(n_splits=3, purge_gap=purge_gap, random_state=config['model']['random_state'])
        best_model = trainer.train(X_train_final, y_train_final)
        
        print(f"  -> Best Params: depth={best_model.max_depth}, lr={best_model.learning_rate}")

        # Sinh tín hiệu: Lọc ngưỡng xác suất -> Đảo chiều tín hiệu -> GMM Hard Filter[cite: 1, 2, 3]
        sig_gen = SignalGenerator(confidence_threshold=config['model']['confidence_threshold'])
        fold_result_df = sig_gen.generate(best_model, X_test_final)
        fold_result_df['fold'] = fold + 1
        fold_result_df['actual_target'] = y_test_final

        oos_predictions.append(fold_result_df)
        trained_models.append(best_model)

    # -------------------------------------------------------------------------
    # BACKTEST & HIỆU SUẤT ĐẦU TƯ THỰC TẾ
    # -------------------------------------------------------------------------
    full_oos_predictions = pd.concat(oos_predictions).sort_index()
    print("\n" + "=" * 80)
    print(f"HOÀN TẤT PREDICTION TRÊN {len(full_oos_predictions)} SAMPLES OOS")
    print("=" * 80)

    simulator = VectorizedBacktestSimulator(
        fee_bps=config['backtest']['fee_bps'], 
        slippage_bps=config['backtest']['slippage_bps'], 
        execution_lag=config['backtest']['execution_lag']
    )

    print("\nOUT-OF-SAMPLE BACKTEST REPORT (NET OF FEES)[cite: 1, 2, 3]")
    
    metrics_list = []
    for fd in sorted(full_oos_predictions['fold'].unique()):
        fold_preds = full_oos_predictions[full_oos_predictions['fold'] == fd]['predicted_target']
        metrics = simulator.run(fold_preds, df_final)
        metrics['Fold'] = f"Fold_{fd}"
        metrics_list.append(metrics)

    # Tổng hợp toàn bộ quá trình OOS (Overall)
    overall_metrics = simulator.run(full_oos_predictions['predicted_target'], df_final)
    overall_metrics['Fold'] = "OVERALL"
    metrics_list.append(overall_metrics)

    report_df = pd.DataFrame(metrics_list).set_index('Fold')
    report_df = report_df.drop(columns=['Equity_Curve'])
    
    print(report_df.round(4).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline Quantitative Research 5-Stage")
    parser.add_argument("--config", type=str, default="configs/asset_config.yaml", help="Đường dẫn đến file cấu hình YAML")
    args = parser.parse_args()
    
    main(args.config)