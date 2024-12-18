#include <cassert>
#include <gtest/gtest.h>

#include <console_bridge/console.h>

//////////////////////////////////////////////////
TEST(ConsoleTest, MacroExpansionTest_ItShouldCompile)
{
  if (true)
    CONSOLE_BRIDGE_logDebug("Testing Log");

  if (true)
    CONSOLE_BRIDGE_logDebug("Testing Log");
  else
  {
      assert(true);
  }

  if (true)
  {
    CONSOLE_BRIDGE_logDebug("Testing Log");
  }
  else
  {
    CONSOLE_BRIDGE_logDebug("Testing Log");
  }
}
