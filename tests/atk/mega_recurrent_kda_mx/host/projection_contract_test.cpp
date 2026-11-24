#include "projection_contract.h"

#include <cassert>

int main()
{
    using namespace mega_recurrent_kda_mx_contract;
    static_assert(ValidProjectionK(64) && ValidProjectionK(7168) && ValidProjectionK(65536));
    static_assert(!ValidProjectionK(-64) && !ValidProjectionK(0) && !ValidProjectionK(32));
    static_assert(!ValidProjectionK(7167) && !ValidProjectionK(65600));
    static_assert(ScalePairs(7168) == 112);
    static_assert(QkvWorkspaceBytes(128, 12) == 1179648);
    assert(QkvWorkspaceBytes(8, 12) == 73728);
    return 0;
}
