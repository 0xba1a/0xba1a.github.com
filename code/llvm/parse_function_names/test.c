#include <stdio.h>

int add(int a, int b) {
	return a + b;
}

int sub (int a, int b) {
	return a - b;
}

int main() {
	int a = 10;
	int b = 20;

	printf("Addition: %d\n", add(a, b));
	printf("Subtraction: %d\n", sub(a, b));

	return 0;
}
