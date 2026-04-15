**ML Based CPU Scheduler**
A Comparative Study for Optimized Process Execution

#Group Members:
1. Imdadul Sk (CS2511)
2. Himanshu Kushwaha (CS2510)


#Problem Statement:
In modern Operating Systems, the CPU scheduler must make split-second decisions to manage ever-increasing computational complexity. Traditional, static scheduling strategies often fail to balance throughput, response time, and energy efficiency in high-demand environments.


#Solution:
This project investigates the integration of Machine Learning into CPU scheduling to move beyond fixed-policy constraints. By comparing **Support Vector Machines (SVM)**, **Logistic Regression**, **SGD Classifiers**, and **Random Forest**, this system dynamically predicts the most effective scheduling method for incoming process requests. The goal is to optimize critical performance metrics—including "Turnaround Time", "Throughput", and "Waiting Time".


#The Process Life Cycle:
The scheduler manages processes by simulating the standard state transitions found in uniprocessor systems:
1. **New**: New process requests are initialized with specific Burst Times and Priorities.
2. **Ready**: Processes are held in a queue, awaiting a scheduling decision.
3. **Running**: Instead of a static rule, the Best-Performing ML Model predicts the strategy (FCFS, SJF, Priority, or RR) that will yield the quickest turnaround time for that specific request.
4. **Waiting**: The system supports preemption, allowing higher-priority tasks to interrupt lower-priority execution, ensuring high responsiveness.
5. **Terminated**: Once execution is complete, the process lifecycle is visualized on a Gantt Chart.


#Why Compare Multiple ML Models?
The focus of this project is to determine which mathematical approach best handles the complexity of modern scheduling. Since different machine learning models respond uniquely to different data distributions, a comparative approach is essential to find the optimal solution.

1. Data-Model Compatibility: Different datasets have different underlying patterns. By comparing multiple models, we can identify which architecture (e.g., the decision branches of Random Forest vs. the hyperplanes of SVM) best captures the relationship between process features and execution efficiency.
2. Minimizing Turnaround Time: Our primary objective is to predict the "winning" scheduling algorithm with high precision. Selecting the best-performing model ensures that processes are assigned the strategy that yields the lowest possible turnaround time, directly improving system responsiveness.
3. Linear vs. Non-Linear Dynamics: We test if simpler, high-speed models (Logistic Regression) can compete with complex ensemble methods (Random Forest) in predicting scheduling winners.
4. Resource Optimization: Efficient scheduling minimizes CPU idle time and unnecessary context switching. This optimization reduces energy consumption.


