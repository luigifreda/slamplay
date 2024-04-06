#include <iostream>
#include <string>

#include <rerun.hpp>

/*
Launch this with
echo 'hello from stdin!' | ./build/test_rerun_stdio | rerun -

*/
int main() {
    const auto rec = rerun::RecordingStream("rerun_example_stdio");
    rec.to_stdout().exit_on_failure();

    std::string input;
    std::string line;
    while (std::getline(std::cin, line)) {
        input += line + '\n';
    }

    rec.log("stdin", rerun::TextDocument(input));

    return 0;
}
