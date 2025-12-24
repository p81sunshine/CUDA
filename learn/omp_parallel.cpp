#include <iostream>
#include <omp.h>

int main() {
    const int n = 200;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp critical
        {
            std::cout << "迭代 " << i << " 由线程 " << omp_get_thread_num() << "执行" << std::endl;
        }
    }
    
    return 0;
}
