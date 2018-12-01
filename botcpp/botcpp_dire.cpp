// g++ -shared -o botcpp_dire.so -fPIC botcpp_dire.cpp
// g++ -shared -o botcpp_dire.so -fPIC botcpp_dire.cpp -std=c++11 -lprotobuf
//g++ -shared -o botcpp_dire.so -fPIC botcpp_dire.cpp dota_gcmessages_common_bot_script.pb.cc -std=c++11 -lprotobuf && cp *.so /Users/tzaman/Library/Application\ Support/Steam/SteamApps/common/dota\ 2\ beta/game/dota/scripts/vscripts/bots
#include <iostream>
#include "dota_gcmessages_common_bot_script.pb.h"

extern "C" void Init(void * a, void * b) {
    // This is run during Dota2's LoadScript routine, after searching for the lua files.
    std::cout << "Init:: cout" << '\n';
    std::cerr << "Init:: cerr" << '\n';
    // return 1;
}
extern "C" void Observe(const CMsgBotWorldState * state) {
    std::cout << "Observe:: cout" << '\n';
    std::cerr << "Observe:: cerr" << '\n';
    // exit(1);
    // return 
}
extern "C" void Act() {
    std::cout << "Act::" << '\n';
    // exit(1);
}
extern "C" void Shutdown() {
    std::cout << "Shutdown::" << '\n';
    // exit(1);
}
