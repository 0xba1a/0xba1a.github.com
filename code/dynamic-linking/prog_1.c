#include <stdio.h>
#include "my_math.h"

int main() {
    int a, b;
    printf("Enter two integers: ");
    scanf("%d %d", &a, &b);
    
    printf("Sum: %d\n", add(a, b));
    
    return 0;
}