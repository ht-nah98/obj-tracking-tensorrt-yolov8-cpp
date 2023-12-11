#include <iostream>
#include <vector>
#include <cmath>
#include <map>

class EuclideanDistTracker {
private:
    // Store the center positions of the objects
    std::map<int, std::pair<int, int>> center_points;
    // each time a new object id detected, the count will increase by one
    int id_count;

public:
    EuclideanDistTracker() : id_count(0) {}

    std::vector<std::vector<int>> update(const std::vector<std::vector<int>>& objects_rect) {
        // Objects boxes and ids
        std::vector<std::vector<int>> objects_bbs_ids;

        // Get center point of new object
        for (const auto& rect : objects_rect) {
            int x = rect[0];
            int y = rect[1];
            int w = rect[2];
            int h = rect[3];
            int label = rect[4];
            int cx = (x + x + w) / 2;
            int cy = (y + y + h) / 2;

            // Find out if that object was detected already
            bool same_object_detected = false;
            for (const auto& entry : center_points) {
                int id = entry.first;
                const auto& pt = entry.second;
                double dist = std::hypot(cx - pt.first, cy - pt.second);

                if (dist < 25) {
                    center_points[id] = std::make_pair(cx, cy);
                    // std::cout << "Object ID " << id << " updated at (" << cx << ", " << cy << ")\n";
                    objects_bbs_ids.push_back({x, y, w, h, id, label});
                    same_object_detected = true;
                    break;
                }
            }

            // New object is detected, assign the ID to that object
            if (!same_object_detected) {
                center_points[id_count] = std::make_pair(cx, cy);
                objects_bbs_ids.push_back({x, y, w, h,id_count, label});
                // std::cout << "New Object ID " << id_count << " detected at (" << cx << ", " << cy << ")\n";
                id_count++;
            }
        }

        // Clean the dictionary by center points to remove IDS not used anymore
        std::map<int, std::pair<int, int>> new_center_points;
        for (const auto& obj_bb_id : objects_bbs_ids) {
            int object_id = obj_bb_id[4];
            const auto& center = center_points[object_id];
            new_center_points[object_id] = center;
        }

        // Update dictionary with IDs not used removed
        center_points = new_center_points;
        return objects_bbs_ids;
    }
};

