"""
Pose publisher utilities.
The standard real-time deployment uses RealtimeLocalizer's queue-based publish approach.
This file can be extended for ROS or DDS based publishing if needed.
"""
def publish_pose(pose_queue, pose):
    """Publish to queue, dropping the oldest if full."""
    import queue
    try:
        pose_queue.get_nowait()
    except queue.Empty:
        pass
    pose_queue.put_nowait(pose)
