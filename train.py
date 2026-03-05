import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ========================================
# 1. LOAD DATA & CREATE SPLITS
# ========================================
print("Loading LFW dataset...")
lfw = fetch_lfw_people(min_faces_per_person=50, resize=0.4, color=False)
X = lfw.data.astype(np.float32)
y = lfw.target
target_names = lfw.target_names
h, w = lfw.images.shape[1:3]

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(target_names)}")
print(f"Classes: {list(target_names)}")
print()

# Create 60/15/25 train/val/test split (stratified)
X_trval, X_te, y_trval, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_trval, y_trval, test_size=0.20, stratify=y_trval, random_state=RANDOM_STATE
)

print(f"Train size: {X_tr.shape[0]}")
print(f"Val size: {X_val.shape[0]}")
print(f"Test size: {X_te.shape[0]}")
print()

# ========================================
# 2. LINEAR SVM - TUNE C ON VALIDATION
# ========================================
print("=" * 60)
print("LINEAR SVM - Tuning C on validation set")
print("=" * 60)

C_grid_linear = [0.01, 0.1, 1, 10, 100]
best_C_linear = None
best_val_acc_linear = 0

results_linear = []

for C in C_grid_linear:
    # Build pipeline with StandardScaler and Linear SVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=C, random_state=RANDOM_STATE))
    ])
    
    # Fit on training data
    pipeline.fit(X_tr, y_tr)
    
    # Predict on validation set
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    results_linear.append({'C': C, 'val_acc': val_acc})
    print(f"C={C:>6g} | Val Acc={val_acc:.4f}")
    
    if val_acc > best_val_acc_linear:
        best_val_acc_linear = val_acc
        best_C_linear = C

print(f"\nBest C for Linear SVM: {best_C_linear} (Val Acc={best_val_acc_linear:.4f})")
print()

# Retrain on train+val with best C and evaluate on test
print("Retraining Linear SVM on train+val with best C...")
X_trval_combined = np.vstack([X_tr, X_val])
y_trval_combined = np.hstack([y_tr, y_val])

pipeline_linear_best = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', C=best_C_linear, random_state=RANDOM_STATE))
])
pipeline_linear_best.fit(X_trval_combined, y_trval_combined)
y_te_pred_linear = pipeline_linear_best.predict(X_te)
test_acc_linear = accuracy_score(y_te, y_te_pred_linear)

print(f"Linear SVM Test Accuracy: {test_acc_linear:.4f}")
print()

# ========================================
# 3. RBF SVM - TUNE C AND GAMMA ON VALIDATION
# ========================================
print("=" * 60)
print("RBF SVM - Tuning C and gamma on validation set")
print("=" * 60)

C_grid_rbf = [0.1, 1, 10, 100]
gamma_grid = ['scale', 0.001, 0.01, 0.1]

best_C_rbf = None
best_gamma_rbf = None
best_val_acc_rbf = 0

results_rbf = []

for C in C_grid_rbf:
    for gamma in gamma_grid:
        # Build pipeline with StandardScaler and RBF SVC
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=C, gamma=gamma, random_state=RANDOM_STATE))
        ])
        
        # Fit on training data
        pipeline.fit(X_tr, y_tr)
        
        # Predict on validation set
        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        results_rbf.append({'C': C, 'gamma': gamma, 'val_acc': val_acc})
        print(f"C={C:>6g}, gamma={str(gamma):>8s} | Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc_rbf:
            best_val_acc_rbf = val_acc
            best_C_rbf = C
            best_gamma_rbf = gamma

print(f"\nBest params for RBF SVM: C={best_C_rbf}, gamma={best_gamma_rbf} (Val Acc={best_val_acc_rbf:.4f})")
print()

# Retrain on train+val with best params and evaluate on test
print("Retraining RBF SVM on train+val with best params...")
pipeline_rbf_best = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=best_C_rbf, gamma=best_gamma_rbf, random_state=RANDOM_STATE))
])
pipeline_rbf_best.fit(X_trval_combined, y_trval_combined)
y_te_pred_rbf = pipeline_rbf_best.predict(X_te)
test_acc_rbf = accuracy_score(y_te, y_te_pred_rbf)

print(f"RBF SVM Test Accuracy: {test_acc_rbf:.4f}")
print()

# ========================================
# 4. OPTIONAL: PCA + SVM
# ========================================
print("=" * 60)
print("OPTIONAL: PCA + SVM")
print("=" * 60)

variance_ratios = [0.90, 0.95, 0.99]
results_pca = []

for var_ratio in variance_ratios:
    print(f"\nTesting PCA with {var_ratio*100:.0f}% variance retained...")
    
    # Determine number of components needed
    pca_temp = PCA(n_components=var_ratio, svd_solver='full', random_state=RANDOM_STATE)
    pca_temp.fit(StandardScaler().fit_transform(X_tr))
    n_components = pca_temp.n_components_
    print(f"  Number of components: {n_components}")
    
    # Test Linear SVM with PCA
    best_C_pca_linear = None
    best_val_acc_pca_linear = 0
    
    for C in C_grid_linear:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='full', random_state=RANDOM_STATE)),
            ('svc', SVC(kernel='linear', C=C, random_state=RANDOM_STATE))
        ])
        pipeline.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, pipeline.predict(X_val))
        
        if val_acc > best_val_acc_pca_linear:
            best_val_acc_pca_linear = val_acc
            best_C_pca_linear = C
    
    # Retrain and test Linear SVM with PCA
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, svd_solver='full', random_state=RANDOM_STATE)),
        ('svc', SVC(kernel='linear', C=best_C_pca_linear, random_state=RANDOM_STATE))
    ])
    pipeline_final.fit(X_trval_combined, y_trval_combined)
    test_acc_pca_linear = accuracy_score(y_te, pipeline_final.predict(X_te))
    
    print(f"  PCA + Linear SVM: Best C={best_C_pca_linear}, Test Acc={test_acc_pca_linear:.4f}")
    
    # Test RBF SVM with PCA
    best_C_pca_rbf = None
    best_gamma_pca_rbf = None
    best_val_acc_pca_rbf = 0
    
    for C in C_grid_rbf:
        for gamma in gamma_grid:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components, svd_solver='full', random_state=RANDOM_STATE)),
                ('svc', SVC(kernel='rbf', C=C, gamma=gamma, random_state=RANDOM_STATE))
            ])
            pipeline.fit(X_tr, y_tr)
            val_acc = accuracy_score(y_val, pipeline.predict(X_val))
            
            if val_acc > best_val_acc_pca_rbf:
                best_val_acc_pca_rbf = val_acc
                best_C_pca_rbf = C
                best_gamma_pca_rbf = gamma
    
    # Retrain and test RBF SVM with PCA
    pipeline_final_rbf = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, svd_solver='full', random_state=RANDOM_STATE)),
        ('svc', SVC(kernel='rbf', C=best_C_pca_rbf, gamma=best_gamma_pca_rbf, random_state=RANDOM_STATE))
    ])
    pipeline_final_rbf.fit(X_trval_combined, y_trval_combined)
    test_acc_pca_rbf = accuracy_score(y_te, pipeline_final_rbf.predict(X_te))
    
    print(f"  PCA + RBF SVM: Best C={best_C_pca_rbf}, gamma={best_gamma_pca_rbf}, Test Acc={test_acc_pca_rbf:.4f}")
    
    results_pca.append({
        'var_ratio': var_ratio,
        'n_components': n_components,
        'linear_C': best_C_pca_linear,
        'linear_test_acc': test_acc_pca_linear,
        'rbf_C': best_C_pca_rbf,
        'rbf_gamma': best_gamma_pca_rbf,
        'rbf_test_acc': test_acc_pca_rbf
    })

print()

# ========================================
# 5. GENERATE CONFUSION MATRICES
# ========================================
print("=" * 60)
print("GENERATING CONFUSION MATRICES")
print("=" * 60)

# Confusion matrix for Linear SVM
cm_linear = confusion_matrix(y_te, y_te_pred_linear)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_linear, display_labels=target_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title(f'Linear SVM Confusion Matrix (Test Acc={test_acc_linear:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix_linear.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_linear.png")

# Confusion matrix for RBF SVM
cm_rbf = confusion_matrix(y_te, y_te_pred_rbf)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=target_names)
disp.plot(ax=ax, cmap='Greens', xticks_rotation=45)
plt.title(f'RBF SVM Confusion Matrix (Test Acc={test_acc_rbf:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix_rbf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_rbf.png")

# ========================================
# 6. SUMMARY OF RESULTS
# ========================================
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Linear SVM:")
print(f"  Best C: {best_C_linear}")
print(f"  Test Accuracy: {test_acc_linear:.4f}")
print()
print(f"RBF SVM:")
print(f"  Best C: {best_C_rbf}")
print(f"  Best gamma: {best_gamma_rbf}")
print(f"  Test Accuracy: {test_acc_rbf:.4f}")
print()

if results_pca:
    print("PCA + SVM Results:")
    for res in results_pca:
        print(f"\n  Variance Ratio: {res['var_ratio']*100:.0f}% (k={res['n_components']})")
        print(f"    Linear: C={res['linear_C']}, Test Acc={res['linear_test_acc']:.4f}")
        print(f"    RBF: C={res['rbf_C']}, gamma={res['rbf_gamma']}, Test Acc={res['rbf_test_acc']:.4f}")

print()
print("Script completed successfully!")
