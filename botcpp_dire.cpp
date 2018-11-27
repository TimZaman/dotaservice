// g++ -shared -o botcpp_dire.so -fPIC botcpp_dire.cpp
#include <iostream>

extern "C" void Init() {
    // This is run during Dota2's LoadScript routine, after searching for the lua files.
    std::cout << "Init::" << '\n';
}
extern "C" void Observe() {
    std::cout << "Observe::" << '\n';
    exit(0);
}
extern "C" void Act() {
    std::cout << "Act::" << '\n';
    exit(0);
}
extern "C" void Shutdown() {
    std::cout << "Shutdown::" << '\n';
    exit(0);
}
