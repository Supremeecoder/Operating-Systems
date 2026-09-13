# ML Based CPU Scheduler: A Comparative Study for Optimized Process Execution

**Group Members:**
1. Himanshu Kushwaha (CS2510)
2. Imdadul Sk (CS2511)


## Problem Statement

In modern Operating Systems, the CPU scheduler must make split-second decisions to manage ever-increasing computational complexity. Traditional, static scheduling strategies often fail to balance throughput, response time, and energy efficiency in high-demand environments.

## Solution

This project investigates the integration of Machine Learning into CPU scheduling to move beyond fixed-policy constraints. By comparing **Support Vector Machines (SVM)**, **Logistic Regression**, **SGD Classifiers**, and **Random Forest**, this system dynamically predicts the most effective scheduling method for incoming process requests. The goal is to optimize critical performance metrics — including Turnaround Time, Throughput, and Waiting Time.

Model selection is done using **5-fold cross-validation on the training set only**, so the reported accuracy of the chosen model is an honest, unbiased estimate rather than a number obtained by peeking at the test set. On our dataset, **Random Forest** was selected as the best-performing model, reaching ~93% cross-validated accuracy and ~95.5% accuracy on the held-out test set.

## The Process Life Cycle

The scheduler manages processes by simulating the standard state transitions found in uniprocessor systems:

1. **New:** New process requests are initialized with specific Burst Time, Priority, Arrival Time, Time Quantum, and Resource requirements.
2. **Ready:** Processes are held in a queue, awaiting a scheduling decision.
3. **Running:** Instead of a static rule, the best-performing ML model predicts the strategy (FCFS, SJF, Priority, or RR) that would have yielded the quickest turnaround time for that specific process. Each process is then actually dispatched under **its own predicted algorithm** in the simulation — FCFS/SJF run to completion once started, Priority can preempt, and RR is sliced by time quantum.
4. **Waiting:** The system supports preemption: a running Priority-scheduled process is interrupted if a process with a strictly higher priority becomes ready, and RR processes are returned to the back of the queue when their quantum expires. Since the ready queue can contain processes running under different predicted algorithms at once, the next process to run is chosen using a normalized ranking (each process is compared using the metric its *own* algorithm cares about — arrival time for FCFS, remaining burst for SJF, priority number for Priority, queue-wait time for RR).
5. **Terminated:** Once execution is complete, the full process lifecycle is visualized on a Gantt Chart, with each block labeled by the process ID and the algorithm it was scheduled under.

## Why Compare Multiple ML Models?

The focus of this project is to determine which mathematical approach best handles the complexity of modern scheduling. Since different machine learning models respond uniquely to different data distributions, a comparative approach is essential to find the optimal solution.

1. **Data-Model Compatibility:** Different datasets have different underlying patterns. By comparing multiple models, we can identify which architecture (e.g., the decision branches of Random Forest vs. the hyperplanes of SVM) best captures the relationship between process features and execution efficiency.
2. **Minimizing Turnaround Time:** Our primary objective is to predict the "winning" scheduling algorithm with high precision. Selecting the best-performing model ensures that processes are assigned the strategy that yields the lowest possible turnaround time, directly improving system responsiveness.
3. **Linear vs. Non-Linear Dynamics:** We test whether simpler, high-speed models (Logistic Regression) can compete with complex ensemble methods (Random Forest) in predicting scheduling winners.
4. **Resource Optimization:** Efficient scheduling minimizes CPU idle time and unnecessary context switching, reducing energy consumption.

## Methodology Notes

A few implementation details worth documenting, since they affect how the results should be interpreted:

- **Missing-value handling:** Burst Time and Resources are imputed (mean/median) *before* the scheduling-algorithm labels are generated, so the label-generation step never operates on missing data.
- **Label noise:** ~8% of training labels are intentionally randomized to simulate real-world unpredictability in scheduling outcomes. This caps the theoretical maximum achievable accuracy at roughly 94% (1 − noise rate + noise rate / number of classes), so accuracy near that figure reflects the data design, not a model deficiency.
- **Avoiding data leakage:** All four models are wrapped in a `Pipeline(StandardScaler, classifier)` so that feature scaling is refit independently within every cross-validation fold, and the test set is used exactly once, for the final reported accuracy of the selected model.
- **Reproducibility:** All stochastic models (SVM, Logistic Regression, SGD Classifier, Random Forest) use a fixed random seed, so results are consistent across runs.
- **Class balance:** `class_weight='balanced'` is applied consistently across all four models to account for the uneven distribution of scheduling-algorithm labels.

## Results Summary

| Model | 5-fold CV Accuracy | Held-out Test Accuracy |
|---|---|---|
| SVM | ~91% | ~93% |
| Logistic Regression | ~84% | ~87% |
| SGD Classifier | ~87% | ~90% |
| **Random Forest (selected)** | **~93%** | **~95.5%** |

*(Exact figures vary slightly by dataset file used and random seed.)*
