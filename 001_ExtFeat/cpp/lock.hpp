#ifndef LOCK_HPP
#define LOCK_HPP

#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

/**
 * Function to lock the access to file. Return true if previously unlocked and now able to
 * lock
 */
bool lock(fs::path fpath) {
    fs::path lock_fpath = fs::path(fpath.string() + ".lock");
    if (fs::exists(fpath) || fs::exists(lock_fpath)) return false;
    try {
      fs::create_directories(lock_fpath);
    } catch(const boost::filesystem::filesystem_error& e) {}
    return true;
}

bool unlock(fs::path fpath) {
    fs::path lock_fpath = fs::path(fpath.string() + ".lock");
    if (!fs::exists(lock_fpath)) return false;
    try {
      return fs::remove(lock_fpath);
    } catch(const boost::filesystem::filesystem_error& e) {}
}

#endif

