#include "ByteTrack/STrack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iomanip>

byte_track::STrack::STrack(const Rect<float>& rect, const float& score) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0),
    peak_vel_x_(0.0f),
    peak_vel_y_(0.0f)
{
}

byte_track::STrack::~STrack()
{
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}
const float& byte_track::STrack::getScore() const
{
    return score_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    if (frame_id == 1)
    {
        is_activated_ = true;
    }
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::predict()
{
    predict(1.0f);  // Default dt=1.0
}

void byte_track::STrack::predict(float dt)
{
    // Store pre-predict values for logging
    float pre_x = mean_[0];
    float vel_x = mean_[4];
    float vel_y = mean_[5];
    
    // Cap dt for ALL tracks to avoid overshooting with variable frame rate
    float effective_dt = std::min(dt, 2.0f);  // Max 2x normal frame movement for tracked
    
    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;  // Reset height velocity for non-tracked
        
        // Even stricter cap for lost tracks
        effective_dt = std::min(dt, 1.5f);  // Max 1.5x normal frame movement
        
        // Use peak velocity if current velocity has decayed too much
        float peak_speed = std::sqrt(peak_vel_x_ * peak_vel_x_ + peak_vel_y_ * peak_vel_y_);
        float curr_speed = std::sqrt(mean_[4] * mean_[4] + mean_[5] * mean_[5]);
        if (peak_speed > 0.1f && curr_speed < peak_speed * 0.5f)
        {
            mean_[4] = peak_vel_x_;
            mean_[5] = peak_vel_y_;
        }
    }
    
    kalman_filter_.predict(mean_, covariance_, effective_dt);
    updateRect();
    
    // Debug: show prediction details for tracks with significant velocity
    if (std::abs(vel_x) > 2.0f || std::abs(vel_y) > 2.0f)
    {
        float delta_x = mean_[0] - pre_x;
        float expected_delta = vel_x * effective_dt;
        std::string dt_info = (effective_dt != dt) ? 
            " (capped from " + std::to_string(dt).substr(0,4) + ")" : "";
        std::cout << "    [KF] id=" << track_id_ 
                  << " dt=" << std::fixed << std::setprecision(2) << effective_dt << dt_info
                  << " vel_x=" << std::setprecision(1) << vel_x 
                  << " expected_dx=" << expected_delta 
                  << " actual_dx=" << delta_x << std::endl;
    }
}

void byte_track::STrack::update(const STrack &new_track, const size_t &frame_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();
    updatePeakVelocity();  // Track peak velocity after update

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;
}

std::pair<float, float> byte_track::STrack::getVelocity() const
{
    return std::make_pair(mean_[4], mean_[5]);
}

void byte_track::STrack::updatePeakVelocity()
{
    float curr_speed = std::sqrt(mean_[4] * mean_[4] + mean_[5] * mean_[5]);
    float peak_speed = std::sqrt(peak_vel_x_ * peak_vel_x_ + peak_vel_y_ * peak_vel_y_);
    
    if (curr_speed > peak_speed)
    {
        peak_vel_x_ = mean_[4];
        peak_vel_y_ = mean_[5];
    }
}

void byte_track::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateRect()
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;
}
