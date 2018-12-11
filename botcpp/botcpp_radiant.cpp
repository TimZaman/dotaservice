//g++ -shared -o botcpp_radiant.so -fPIC botcpp_radiant.cpp dota_gcmessages_common_bot_script.pb.cc -std=c++11 -lprotobuf && cp *.so /Users/tzaman/Library/Application\ Support/Steam/SteamApps/common/dota\ 2\ beta/game/dota/scripts/vscripts/bots
#include <chrono> 
#include <ctime> 
#include <iostream>
#include <thread>

#include <google/protobuf/text_format.h>

#include "dota_gcmessages_common_bot_script.pb.h"


using namespace std;


extern "C" void Init(void * a, void *b, void *c) {
    // This is run during Dota2's LoadScript routine, after searching for the lua files.
    cout << "Init::cout" << endl;
}


extern "C" void Observe(void * a, void *b, void *c) {
    // The intention of this script is most probably to setup a connection with the
    // worldstate sockets. Kinda weird though, because the dota game would be the server
    // and then this script would be the client of that..
    // Observe is called every tick that corresponds to the worldstate server's ticks,
    // Exactly before the Act() is called.
    cout << "Observe::cout" << endl;
}

int i = -1;

// Returning std::shared_ptr<CMsgBotWorldState::Actions> crashes upon return.
// Returning CMsgBotWorldState::Actions craches.
// void, void *, char* is fine, even if its contents are totally bogus.
extern "C" void * Act() {
    // Act seems to be called practically _exactly_ after Observe is called.
    // Since it is called once per team, all team-decisions need to be made here. That means
    // that we need to communicate all actions. Probably that means we need to return the actions
    // protobuf somehow. I think returning the protobuffer itself, from this function makes
    // the most sense.
    // This call is fully blocking the entire game, so possible they are indeed waiting for a 
    // synchronous return.
    auto timenow =  chrono::system_clock::to_time_t(chrono::system_clock::now()); 
    cout << "Act::cout @" << ctime(&timenow) << endl;

    

    // std::this_thread::sleep_for (std::chrono::seconds(2));

    // std::string data = "actionType: DOTA_UNIT_ORDER_NONE";
 
    // std::string data = "actionType: DOTA_UNIT_ORDER_MOVE_TO_POSITION\n"
    // "moveToLocation {\n"
    // "  location {\n"
    // "      x: 0.0\n"
    // "      y: 0.0\n"
    // "      z: 0.0\n"
    // "  }\n"
    // "}\n";

    std::string data = "actions {\n"
    "  actionType: DOTA_UNIT_ORDER_MOVE_TO_POSITION\n"
    "  player: " + to_string(i) + "\n"
    "  moveToLocation {\n"
    "    location {\n"
    "      x: 0.0\n"
    "      y: 0.0\n"
    "      z: 0.0\n"
    "    }\n"
    "  }\n"
    "}\n";
    CMsgBotWorldState::Actions msg;
    i++;
    if (i > 10) {
        i = -1;
    }

    // bool parsed = msg.ParseFromString(data);
    // cout << "parsed=" << parsed << endl;
    // msg.set_actiontype(CMsgBotWorldState_Action_Type_DOTA_UNIT_ORDER_NONE);

    // CMsgBotWorldState::Actions msg;

    // msg.set_has_movetolocation();
    if (!google::protobuf::TextFormat::ParseFromString(data, &msg)){
        cerr << std::endl << "Failed to parse file!" << endl;
        // return -1;
    }

    std::string s; 
    if (google::protobuf::TextFormat::PrintToString(msg, &s)) {
        std::cout << "Your message:\n" << s;
    } else {
        std::cerr << "Message not valid (partial content: " << msg.ShortDebugString() << ")\n";
    }
    
    // std::cout << "worldstate=" << msg.DebugString() << endl;
    cout << "Gonna return from act..\n";
    // return &msg;
    // return "foobar hahaha hihih";
    std::string output;
    msg.SerializeToString(&output);
    // output = data;
    cout << "out len=" << output.size() << " contents=" << output << endl;

    char * cc = new char [output.length()+1];
    strcpy (cc, output.c_str());

    return cc;
    // return &msg;
    // std::shared_ptr<CMsgBotWorldState::Actions> smsg = std::make_shared<CMsgBotWorldState::Actions>(msg);
    // return smsg;
}

extern "C" void Shutdown() {
    // This is simply called when the game is over.
    std::cout << "Shutdown::cout" << '\n';
}
