import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

df = pd.read_csv('DataSet3.csv')

def calculate_tat(row):
    bt = row['BT']
    priority = row['Priority']
    quantum = row['TimeQuantum']

    noise = np.random.normal(0, 32)

    tat_fcfs = bt + noise
    tat_sjf = (bt * 0.8 + noise) if bt < 150 else (bt * 1.2 + noise)
    tat_priority = (bt * 0.75 + noise) if priority < 4 else (bt * 1.3 + noise)
    tat_rr = bt * (0.85 if quantum > 100 else 1.05) + noise

    results = {0: tat_fcfs, 1: tat_sjf, 2: tat_priority, 3: tat_rr}
    true_winner = min(results, key=results.get)

    if np.random.rand() < 0.08:
        return np.random.randint(0, 4)

    return true_winner

df['target'] = df.apply(calculate_tat, axis=1)
print("Target distribution:\n", df['target'].value_counts())


features = ['BT', 'Resources', 'ArrivalTime', 'Priority', 'TimeQuantum']
X = df[features].copy()
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Data Cleaning
bt_mean = X_train['BT'].mean()
res_median = X_train['Resources'].median()

X_train['BT'] = X_train['BT'].fillna(bt_mean)
X_test['BT'] = X_test['BT'].fillna(bt_mean)

X_train['Resources'] = X_train['Resources'].fillna(res_median)
X_test['Resources'] = X_test['Resources'].fillna(res_median)

df['BT'] = df['BT'].fillna(bt_mean)
df['Resources'] = df['Resources'].fillna(res_median)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "SVM": SVC(kernel='rbf', C=10, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SGD Classifier": SGDClassifier(loss='log_loss', eta0=0.01, learning_rate='constant', class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

algo_names = ['FCFS', 'SJF', 'Priority', 'RR']

# Training
fig, axes = plt.subplots(1, len(models), figsize=(12, 5))

best_model = None
best_acc = 0

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=algo_names,
        yticklabels=algo_names,
        linewidths=0.5,
        linecolor='black',
        cbar=True,
        ax=axes[i]
    )

    axes[i].set_title(f"{name}", fontsize=12)
    axes[i].set_xlabel("Predicted Value", fontsize=10) 
    axes[i].set_ylabel("Actual Value", fontsize=10)    

plt.tight_layout()
plt.show()

print(f"\nBest Model: {type(best_model).__name__} with accuracy {best_acc*100:.2f}%")

# Gantt Chart
from collections import deque

plot_df = df.head(10).copy()
plot_df = plot_df.sort_values(by='ArrivalTime').reset_index(drop=True)

test_features = plot_df[features]
test_scaled = scaler.transform(test_features)
plot_preds = best_model.predict(test_scaled)

remaining_bt = plot_df['BT'].tolist()
arrival = plot_df['ArrivalTime'].tolist()
priority = plot_df['Priority'].tolist()
quantum = plot_df['TimeQuantum'].tolist()
job_ids = plot_df.get('Jobid', plot_df.index).tolist()

time = 0
i = 0
queue = deque()
gantt = []
current_process = None

def pick_next(queue):
    best_idx = min(queue, key=lambda x: priority[x])
    queue.remove(best_idx)
    return best_idx

while i < len(plot_df) or queue or current_process is not None:

    while i < len(plot_df) and arrival[i] <= time:
        queue.append(i)
        i += 1

    if current_process is None and queue:
        current_process = pick_next(queue)

    if current_process is None:
        time = arrival[i]
        continue

    idx = current_process
    tq = quantum[idx]
    algo = plot_preds[idx]

    #Time until next process arrives
    next_arrival = arrival[i] if i < len(arrival) else float('inf')
    time_slice = next_arrival - time

    if algo == 3:
        run_time = min(tq, remaining_bt[idx], time_slice)
    elif algo == 2:
        run_time = min(remaining_bt[idx], time_slice)
    else: 
        run_time = remaining_bt[idx]

    start_time = time
    time += run_time
    remaining_bt[idx] -= run_time

    gantt.append((job_ids[idx], start_time, run_time, algo))

    while i < len(plot_df) and arrival[i] <= time:
        queue.append(i)
        i += 1

    if queue:
        highest = min(queue, key=lambda x: priority[x])
        if priority[highest] < priority[idx]:
            queue.append(idx)
            current_process = pick_next(queue)
            continue

    if remaining_bt[idx] > 0 and algo == 3:
        queue.append(idx)

    current_process = None


plt.figure(figsize=(15, 6))

palette = sns.color_palette("Set2", len(job_ids))
job_color_map = {job: palette[i] for i, job in enumerate(job_ids)}

for job, start, duration, pred in gantt:
    plt.broken_barh([(start, duration)], (10, 8),
                    facecolors=job_color_map[job], edgecolor='black')

    plt.text(start + duration/2, 14,
             f"P-{job}\n({algo_names[pred]})",
             ha='center', va='center', fontsize=8)

plt.title(f"Gantt Chart - {type(best_model).__name__}")
plt.xlabel("Time")
plt.yticks([])
plt.grid(axis='x')
plt.show()
