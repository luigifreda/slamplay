// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include <iostream>
#include <fstream>

#include <iomanip>
#include <vector>
#include <deque>
#include <map>
#include <list> 
#include <set> 
#include <unordered_set> 

#include "macros.h"
#include "json.h"

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 


int mainFileReadWrite(int argc, char **argv)
{
    std::string infilename = dataDir + "/example.json";
    std::string outfilename = dataDir + "/example_out.json";    

    // read a JSON file
    std::ifstream i(infilename);
    json j;
    i >> j;
    std::cout << "content of " << infilename << ": " << j.dump() << std::endl; 

    // write prettified JSON to another file
    std::ofstream o(outfilename);
    o << std::setw(4) << j << std::endl;

    return 0; 
}

int mainFileParsing(int argc, char **argv) 
{
    std::string filename = dataDir + "/example.json";
    std::ifstream f(filename);
    json data = json::parse(f);

    std::cout << "content of " << filename << ": " << data.dump() << std::endl; 

    return 0;
}

int mainJsonCreation(int argc, char **argv) 
{
    // Using (raw) string literals and json::parse
    json ex1 = json::parse(R"(
    {
        "pi": 3.141,
        "happy": true
    }
    )");

    // Using user-defined (raw) string literals
    json ex2 = R"(
    {
        "pi": 3.141,
        "happy": true
    }
    )"_json;

    // Using initializer lists
    json ex3 = {
        {"pi", 3.141},
        {"happy", true},
        {"name", "Niels"},
        {"nothing", nullptr},
        {"answer", {
            {"everything", 42}
        }},
        {"list", {1, 0, 2}},
        {"object", {
            {"currency", "USD"},
            {"value", 42.99}
        }}
    };

    return 0;    
}    


int mainFromJson(int argc, char **argv)
{
    // a simple struct to model a person
    struct Person {
        std::string name;
        std::string address;
        int age;
    };

    Person p = {"Ned Flanders", "744 Evergreen Terrace", 60};

    // convert to JSON: copy each value into the JSON object
    json j;
    j["name"] = p.name;
    j["address"] = p.address;
    j["age"] = p.age;

    // ...

    // convert from JSON: copy each value from the JSON object
    Person p2 {
        j["name"].get<std::string>(),
        j["address"].get<std::string>(),
        j["age"].get<int>()
    };

    return 0; 
}

int mainJsonMerge(int argc, char **argv)
{
    // a JSON value
    json j_document = R"({
    "a": "b",
    "c": {
        "d": "e",
        "f": "g"
    }
    })"_json;

    // a patch
    json j_patch = R"({
    "a":"z",
    "c": {
        "f": null
    }
    })"_json;

    // apply the patch
    j_document.merge_patch(j_patch);
    // {
    //  "a": "z",
    //  "c": {
    //    "d": "e"
    //  }
    // }

    return 0; 
}

int mainStlLikeInteraction(int argc, char **argv)
{
    // create an array using push_back
    json j;
    j.push_back("foo");
    j.push_back(1);
    j.push_back(true);

    // also use emplace_back
    j.emplace_back(1.78);

    // iterate the array
    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::cout << *it << '\n';
    }

    // range-based for
    for (auto& element : j) {
        std::cout << element << '\n';
    }

    // getter/setter
    const auto tmp = j[0].get<std::string>();
    j[1] = 42;
    [[maybe_unused]] bool foo = j.at(2);

    // comparison
    j == R"(["foo", 1, true, 1.78])"_json;  // true

    // other stuff
    j.size();     // 4 entries
    j.empty();    // false
    j.type();     // json::value_t::array
    j.clear();    // the array is empty again

    // convenience type checkers
    j.is_null();
    j.is_boolean();
    j.is_number();
    j.is_object();
    j.is_array();
    j.is_string();

    // create an object
    json o;
    o["foo"] = 23;
    o["bar"] = false;
    o["baz"] = 3.141;

    // also use emplace
    o.emplace("weather", "sunny");

    // special iterator member functions for objects
    for (json::iterator it = o.begin(); it != o.end(); ++it) {
        std::cout << it.key() << " : " << it.value() << "\n";
    }

    // the same code as range for
    for (auto& el : o.items()) {
        std::cout << el.key() << " : " << el.value() << "\n";
    }

    // even easier with structured bindings (C++17)
    for (auto& [key, value] : o.items()) {
        std::cout << key << " : " << value << "\n";
    }

    // find an entry
    if (o.contains("foo")) {
        // there is an entry with key "foo"
    }

    // or via find and an iterator
    if (o.find("foo") != o.end()) {
        // there is an entry with key "foo"
    }

    // or simpler using count()
    [[maybe_unused]] int foo_present = o.count("foo"); // 1
    [[maybe_unused]] int fob_present = o.count("fob"); // 0

    // delete an entry
    o.erase("foo");

    return 0; 
}

int mainStlContainerConversion(int argc, char **argv)
{
    std::vector<int> c_vector {1, 2, 3, 4};
    json j_vec(c_vector);
    // [1, 2, 3, 4]

    std::deque<double> c_deque {1.2, 2.3, 3.4, 5.6};
    json j_deque(c_deque);
    // [1.2, 2.3, 3.4, 5.6]

    std::list<bool> c_list {true, true, false, true};
    json j_list(c_list);
    // [true, true, false, true]

    std::forward_list<int64_t> c_flist {12345678909876, 23456789098765, 34567890987654, 45678909876543};
    json j_flist(c_flist);
    // [12345678909876, 23456789098765, 34567890987654, 45678909876543]

    std::array<unsigned long, 4> c_array {{1, 2, 3, 4}};
    json j_array(c_array);
    // [1, 2, 3, 4]

    std::set<std::string> c_set {"one", "two", "three", "four", "one"};
    json j_set(c_set); // only one entry for "one" is used
    // ["four", "one", "three", "two"]

    std::unordered_set<std::string> c_uset {"one", "two", "three", "four", "one"};
    json j_uset(c_uset); // only one entry for "one" is used
    // maybe ["two", "three", "four", "one"]

    std::multiset<std::string> c_mset {"one", "two", "one", "four"};
    json j_mset(c_mset); // both entries for "one" are used
    // maybe ["one", "two", "one", "four"]

    std::unordered_multiset<std::string> c_umset {"one", "two", "one", "four"};
    json j_umset(c_umset); // both entries for "one" are used
    // maybe ["one", "two", "one", "four"]

    return 0;     
}

int mainStlAssociativeContainersConversion(int argc, char **argv)
{
    std::map<std::string, int> c_map { {"one", 1}, {"two", 2}, {"three", 3} };
    json j_map(c_map);
    // {"one": 1, "three": 3, "two": 2 }

    std::unordered_map<const char*, double> c_umap { {"one", 1.2}, {"two", 2.3}, {"three", 3.4} };
    json j_umap(c_umap);
    // {"one": 1.2, "two": 2.3, "three": 3.4}

    std::multimap<std::string, bool> c_mmap { {"one", true}, {"two", true}, {"three", false}, {"three", true} };
    json j_mmap(c_mmap); // only one entry for key "three" is used
    // maybe {"one": true, "two": true, "three": true}

    std::unordered_multimap<std::string, bool> c_ummap { {"one", true}, {"two", true}, {"three", false}, {"three", true} };
    json j_ummap(c_ummap); // only one entry for key "three" is used
    // maybe {"one": true, "two": true, "three": true}
    std::map<std::string, int> c_map2 { {"one", 1}, {"two", 2}, {"three", 3} };
    json j_map2(c_map2);
    // {"one": 1, "three": 3, "two": 2 }

    std::unordered_map<const char*, double> c_umap2 { {"one", 1.2}, {"two", 2.3}, {"three", 3.4} };
    json j_umap2(c_umap2);
    // {"one": 1.2, "two": 2.3, "three": 3.4}

    std::multimap<std::string, bool> c_mmap2 { {"one", true}, {"two", true}, {"three", false}, {"three", true} };
    json j_mmap2(c_mmap2); // only one entry for key "three" is used
    // maybe {"one": true, "two": true, "three": true}

    std::unordered_multimap<std::string, bool> c_ummap2 { {"one", true}, {"two", true}, {"three", false}, {"three", true} };
    json j_ummap2(c_ummap2); // only one entry for key "three" is used
    // maybe {"one": true, "two": true, "three": true}    

    return 0; 
}


int main(int argc, char **argv) 
{
    mainFileReadWrite(argc, argv);    
    mainFileParsing(argc, argv);   
    mainJsonCreation(argc, argv);
    mainFromJson(argc, argv);     
    mainJsonMerge(argc, argv);
    mainStlLikeInteraction(argc, argv);
    mainStlContainerConversion(argc, argv);
    mainStlAssociativeContainersConversion(argc, argv);

    return 0;
}