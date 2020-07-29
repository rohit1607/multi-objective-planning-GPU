#include <stdlib.h>
#include <iostream>

int main(){

    int status;

    status = system("mkdir data/output/test1");
    std::cout << "status = " << status << std::endl;

    status = system("mkdir data/output/test1");
    std::cout << "status = " << status << std::endl;
    return 0;
}
