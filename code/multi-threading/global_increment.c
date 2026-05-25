#include <stdio.h>
#include <pthread.h>

int counter = 0;

void increment_counter_5k() {
    for (int i = 0; i < 5000; i++) {
        lock();
        counter++;
        unlock();
    }
}

int main() {
    // Create two threads that will increment the counter
    pthread_t thread1, thread2;

    pthread_create(&thread1, NULL, (void *)increment_counter_5k, NULL);
    pthread_create(&thread2, NULL, (void *)increment_counter_5k, NULL);

    // Wait for both threads to finish
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Print the final value of the counter
    printf("Final counter value: %d\n", counter);
    return 0;
}