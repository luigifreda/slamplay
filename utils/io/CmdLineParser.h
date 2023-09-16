#pragma once 

#include <string>

// Command line parser
class CmdLineParser
{   int argc; 
    char **argv; 
public: 
    CmdLineParser(int _argc,char **_argv): argc(_argc), argv(_argv){}  

    // check if param exists 
    bool operator[] (std::string param) {
        int idx=-1;  
        for ( int i=0; i<argc && idx==-1; i++ ) 
            if ( std::string ( argv[i] ) == param ) idx=i;    
        return ( idx!=-1 );    
    }

    // get param value, if param does not exists then return provided default value 
    std::string operator()(std::string param, std::string defvalue="-1"){
        int idx=-1;    
        for ( int i=0; i<argc && idx==-1; i++ ) 
            if ( std::string ( argv[i] ) ==param ) idx=i; 
        if ( idx==-1 ) 
            return defvalue;   
        else  
        return ( argv[idx+1] ); 
    }
};
