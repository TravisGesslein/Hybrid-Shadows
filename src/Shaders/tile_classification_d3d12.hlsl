/**********************************************************************
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
RTShadowDenoiser

tile_classification

********************************************************************/

#include "ffx_shadows_dnsr/ffx_denoiser_shadows_util.h"

struct FFX_DNSR_Shadows_Data_Defn
{
    float3   Eye;
    int      FirstFrame;
    int2     BufferDimensions;
    float2   InvBufferDimensions;
    float4x4 ProjectionInverse;
    float4x4 ReprojectionMatrix;
    float4x4 ViewProjectionInverse;
};

cbuffer cbPassData : register(b0)
{
    FFX_DNSR_Shadows_Data_Defn FFX_DNSR_Shadows_Data;
}

Texture2D<float>            t2d_depth              : register(t0);
Texture2D<float2>           t2d_velocity           : register(t1);
Texture2D<float3>           t2d_normal             : register(t2);
Texture2D<float2>           t2d_history            : register(t3);
Texture2D<float>            t2d_previousDepth      : register(t4);
StructuredBuffer<uint>      sb_raytracerResult     : register(t5);


RWStructuredBuffer<uint>    rwsb_tileMetaData             : register(u0);
RWTexture2D<float2>         rwt2d_reprojectionResults     : register(u1);

Texture2D<float3>           t2d_previousMoments    : register(t0, space1);
RWTexture2D<float3>         rwt2d_momentsBuffer           : register(u0, space1);

SamplerState ss_trilinerClamp : register(s0);

float4x4 FFX_DNSR_Shadows_GetViewProjectionInverse()
{
    return FFX_DNSR_Shadows_Data.ViewProjectionInverse;
}

float4x4 FFX_DNSR_Shadows_GetReprojectionMatrix()
{
    return FFX_DNSR_Shadows_Data.ReprojectionMatrix;
}

float4x4 FFX_DNSR_Shadows_GetProjectionInverse()
{
    return FFX_DNSR_Shadows_Data.ProjectionInverse;
}

float2 FFX_DNSR_Shadows_GetInvBufferDimensions()
{
    return FFX_DNSR_Shadows_Data.InvBufferDimensions;
}

int2 FFX_DNSR_Shadows_GetBufferDimensions()
{
    return FFX_DNSR_Shadows_Data.BufferDimensions;
}

int FFX_DNSR_Shadows_IsFirstFrame()
{
    return FFX_DNSR_Shadows_Data.FirstFrame;
}

float3 FFX_DNSR_Shadows_GetEye()
{
    return FFX_DNSR_Shadows_Data.Eye;
}

float FFX_DNSR_Shadows_ReadDepth(int2 p)
{
    return t2d_depth.Load(int3(p, 0)).x;
}

float FFX_DNSR_Shadows_ReadPreviousDepth(int2 p)
{
    return t2d_previousDepth.Load(int3(p, 0)).x;
} 

float3 FFX_DNSR_Shadows_ReadNormals(int2 p)
{
    return normalize(t2d_normal.Load(int3(p, 0)).xyz * 2 - 1.f);
}

float2 FFX_DNSR_Shadows_ReadVelocity(int2 p)
{
    return t2d_velocity.Load(int3(p, 0));
}

float FFX_DNSR_Shadows_ReadHistory(float2 p)
{
    return t2d_history.SampleLevel(ss_trilinerClamp, p, 0).x;
}

float3 FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(int2 p)
{
    return t2d_previousMoments.Load(int3(p, 0)).xyz;
}

uint  FFX_DNSR_Shadows_ReadRaytracedShadowMask(uint p)
{
    return sb_raytracerResult[p];
}

void  FFX_DNSR_Shadows_WriteMetadata(uint p, uint val)
{
    rwsb_tileMetaData[p] = val;
}

void  FFX_DNSR_Shadows_WriteMoments(uint2 p, float3 val)
{
    rwt2d_momentsBuffer[p] = val;
}

void FFX_DNSR_Shadows_WriteReprojectionResults(uint2 p, float2 val)
{
    rwt2d_reprojectionResults[p] = val;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 p)
{
    float depth = FFX_DNSR_Shadows_ReadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

groupshared int g_FFX_DNSR_Shadows_false_count;
bool FFX_DNSR_Shadows_ThreadGroupAllTrue(bool val)
{
    const uint lane_count_in_thread_group = 64;
    if (WaveGetLaneCount() == lane_count_in_thread_group) {
        return WaveActiveAllTrue(val);
    } else {
        GroupMemoryBarrierWithGroupSync();
        g_FFX_DNSR_Shadows_false_count = 0;
        GroupMemoryBarrierWithGroupSync();
        if (!val) g_FFX_DNSR_Shadows_false_count = 1;
        GroupMemoryBarrierWithGroupSync();
        return g_FFX_DNSR_Shadows_false_count == 0;
    }
}

void FFX_DNSR_Shadows_SearchSpatialRegion(uint2 gid, out bool all_in_light, out bool all_in_shadow)
{
    // The spatial passes can reach a total region of 1+2+4 = 7x7 around each block.
    // The masks are 8x4, so we need a larger vertical stride

    // Visualization - each x represents a 4x4 block, xx is one entire 8x4 mask as read from the raytracer result
    // Same for yy, these are the ones we are working on right now

    // xx xx xx
    // xx xx xx
    // xx yy xx <-- yy here is the base_tile below
    // xx yy xx
    // xx xx xx
    // xx xx xx

    // All of this should result in scalar ops
    uint2 base_tile = FFX_DNSR_Shadows_GetTileIndexFromPixelPosition(gid * int2(8, 8));

    // Load the entire region of masks in a scalar fashion
    uint combined_or_mask = 0;
    uint combined_and_mask = 0xFFFFFFFF;
    for (int j = -2; j <= 3; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 tile_index = base_tile + int2(i, j);
            tile_index = clamp(tile_index, 0, int2(FFX_DNSR_Shadows_RoundedDivide(FFX_DNSR_Shadows_GetBufferDimensions().x, 8), FFX_DNSR_Shadows_RoundedDivide(FFX_DNSR_Shadows_GetBufferDimensions().y, 4)) - 1);
            const uint linear_tile_index = FFX_DNSR_Shadows_LinearTileIndex(tile_index, FFX_DNSR_Shadows_GetBufferDimensions().x);
            const uint shadow_mask = FFX_DNSR_Shadows_ReadRaytracedShadowMask(linear_tile_index);

            combined_or_mask = combined_or_mask | shadow_mask;
            combined_and_mask = combined_and_mask & shadow_mask;
        }
    }

    all_in_light = combined_and_mask == 0xFFFFFFFFu;
    all_in_shadow = combined_or_mask == 0u;
}

float FFX_DNSR_Shadows_GetLinearDepth(uint2 did, float depth)
{
    const float2 uv = (did + 0.5f) * FFX_DNSR_Shadows_GetInvBufferDimensions();
    const float2 ndc = 2.0f * float2(uv.x, 1.0f - uv.y) - 1.0f;

    float4 projected = mul(FFX_DNSR_Shadows_GetProjectionInverse(), float4(ndc, depth, 1));
    return abs(projected.z / projected.w);
}

bool FFX_DNSR_Shadows_IsDisoccluded(uint2 did, float depth, float2 velocity)
{
    const int2 dims = FFX_DNSR_Shadows_GetBufferDimensions();
    const float2 texel_size = FFX_DNSR_Shadows_GetInvBufferDimensions();
    const float2 uv = (did + 0.5f) * texel_size;
    const float2 ndc = (2.0f * uv - 1.0f) * float2(1.0f, -1.0f);
    const float2 previous_uv = uv - velocity;

    bool is_disoccluded = true;
    if (all(previous_uv > 0.0) && all(previous_uv < 1.0)) {
        // Read the center values
        float3 normal = FFX_DNSR_Shadows_ReadNormals(did);

        float4 clip_space = mul(FFX_DNSR_Shadows_GetReprojectionMatrix(), float4(ndc, depth, 1.0f));
        clip_space /= clip_space.w; // perspective divide

        // How aligned with the view vector? (the more Z aligned, the higher the depth errors)
        const float4 homogeneous = mul(FFX_DNSR_Shadows_GetViewProjectionInverse(), float4(ndc, depth, 1.0f));
        const float3 world_position = homogeneous.xyz / homogeneous.w;  // perspective divide
        const float3 view_direction = normalize(FFX_DNSR_Shadows_GetEye().xyz - world_position);
        float z_alignment = 1.0f - dot(view_direction, normal);
        z_alignment = pow(z_alignment, 8);

        // Calculate the depth difference
        float linear_depth = FFX_DNSR_Shadows_GetLinearDepth(did, clip_space.z);   // get linear depth

        int2 idx = previous_uv * dims;
        const float previous_depth = FFX_DNSR_Shadows_GetLinearDepth(idx, FFX_DNSR_Shadows_ReadPreviousDepth(idx));
        const float depth_difference = abs(previous_depth - linear_depth) / linear_depth;

        // Resolve into the disocclusion mask
        const float depth_tolerance = lerp(1e-2f, 1e-1f, z_alignment);
        is_disoccluded = depth_difference >= depth_tolerance;
    }

    return is_disoccluded;
}

float2 FFX_DNSR_Shadows_GetClosestVelocity(int2 did, float depth)
{
    float2 closest_velocity = FFX_DNSR_Shadows_ReadVelocity(did);
    float closest_depth = depth;

    float new_depth = QuadReadAcrossX(closest_depth);
    float2 new_velocity = QuadReadAcrossX(closest_velocity);
#ifdef INVERTED_DEPTH_RANGE
    if (new_depth > closest_depth)
#else
    if (new_depth < closest_depth)
#endif
    {
        closest_depth = new_depth;
        closest_velocity = new_velocity;
    }

    new_depth = QuadReadAcrossY(closest_depth);
    new_velocity = QuadReadAcrossY(closest_velocity);
#ifdef INVERTED_DEPTH_RANGE
    if (new_depth > closest_depth)
#else
    if (new_depth < closest_depth)
#endif
    {
        closest_depth = new_depth;
        closest_velocity = new_velocity;
    }

    return closest_velocity * float2(0.5f, -0.5f);  // from ndc to uv
}

#define KERNEL_RADIUS 8
float FFX_DNSR_Shadows_KernelWeight(float i)
{
#define KERNEL_WEIGHT(i) (exp(-3.0 * float(i * i) / ((KERNEL_RADIUS + 1.0) * (KERNEL_RADIUS + 1.0))))

    // Statically initialize kernel_weights_sum
    float kernel_weights_sum = 0;
    kernel_weights_sum += KERNEL_WEIGHT(0);
    for (int c = 1; c <= KERNEL_RADIUS; ++c) {
        kernel_weights_sum += 2 * KERNEL_WEIGHT(c); // Add other half of the kernel to the sum
    }
    float inv_kernel_weights_sum = rcp(kernel_weights_sum);

    // The only runtime code in this function
    return KERNEL_WEIGHT(i) * inv_kernel_weights_sum;
}

void FFX_DNSR_Shadows_AccumulateMoments(float value, float weight, inout float moments)
{
    // We get value from the horizontal neighborhood calculations. Thus, it's both mean and variance due to using one sample per pixel
    moments += value * weight;
}

// The horizontal part of a 17x17 local neighborhood kernel
float FFX_DNSR_Shadows_HorizontalNeighborhood(int2 did)
{
    const int2 base_did = did;

    // Prevent vertical out of bounds access
    if ((base_did.y < 0) || (base_did.y >= FFX_DNSR_Shadows_GetBufferDimensions().y)) return 0;

    const uint2 tile_index = FFX_DNSR_Shadows_GetTileIndexFromPixelPosition(base_did);
    const uint linear_tile_index = FFX_DNSR_Shadows_LinearTileIndex(tile_index, FFX_DNSR_Shadows_GetBufferDimensions().x);

    const int left_tile_index = linear_tile_index - 1;
    const int center_tile_index = linear_tile_index;
    const int right_tile_index = linear_tile_index + 1;

    bool is_first_tile_in_row = tile_index.x == 0;
    bool is_last_tile_in_row = tile_index.x == (FFX_DNSR_Shadows_RoundedDivide(FFX_DNSR_Shadows_GetBufferDimensions().x, 8) - 1);

    uint left_tile = 0;
    if (!is_first_tile_in_row) left_tile = FFX_DNSR_Shadows_ReadRaytracedShadowMask(left_tile_index);
    uint center_tile = FFX_DNSR_Shadows_ReadRaytracedShadowMask(center_tile_index);
    uint right_tile = 0;
    if (!is_last_tile_in_row) right_tile = FFX_DNSR_Shadows_ReadRaytracedShadowMask(right_tile_index);

    // Construct a single uint with the lowest 17bits containing the horizontal part of the local neighborhood.

    // First extract the 8 bits of our row in each of the neighboring tiles
    const uint row_base_index = (did.y % 4) * 8;
    const uint left = (left_tile >> row_base_index) & 0xFF;
    const uint center = (center_tile >> row_base_index) & 0xFF;
    const uint right = (right_tile >> row_base_index) & 0xFF;

    // Combine them into a single mask containting [left, center, right] from least significant to most significant bit
    uint neighborhood = left | (center << 8) | (right << 16);

    // Make sure our pixel is at bit position 9 to get the highest contribution from the filter kernel
    const uint bit_index_in_row = (did.x % 8);
    neighborhood = neighborhood >> bit_index_in_row; // Shift out bits to the right, so the center bit ends up at bit 9.

    float moment = 0.0; // For one sample per pixel this is both, mean and variance

    // First 8 bits up to the center pixel
    uint mask;
    int i;
    for (i = 0; i < 8; ++i) {
        mask = 1u << i;
        moment += (mask & neighborhood) ? FFX_DNSR_Shadows_KernelWeight(8 - i) : 0;
    }

    // Center pixel
    mask = 1u << 8;
    moment += (mask & neighborhood) ? FFX_DNSR_Shadows_KernelWeight(0) : 0;

    // Last 8 bits
    for (i = 1; i <= 8; ++i) {
        mask = 1u << (8 + i);
        moment += (mask & neighborhood) ? FFX_DNSR_Shadows_KernelWeight(i) : 0;
    }

    return moment;
}

groupshared float g_FFX_DNSR_Shadows_neighborhood[8][24];

float FFX_DNSR_Shadows_ComputeLocalNeighborhood(int2 did, int2 gtid)
{
    float local_neighborhood = 0;

    float upper = FFX_DNSR_Shadows_HorizontalNeighborhood(int2(did.x, did.y - 8));
    float center = FFX_DNSR_Shadows_HorizontalNeighborhood(int2(did.x, did.y));
    float lower = FFX_DNSR_Shadows_HorizontalNeighborhood(int2(did.x, did.y + 8));

    g_FFX_DNSR_Shadows_neighborhood[gtid.x][gtid.y] = upper;
    g_FFX_DNSR_Shadows_neighborhood[gtid.x][gtid.y + 8] = center;
    g_FFX_DNSR_Shadows_neighborhood[gtid.x][gtid.y + 16] = lower;

    GroupMemoryBarrierWithGroupSync();

    // First combine the own values.
    // KERNEL_RADIUS pixels up is own upper and KERNEL_RADIUS pixels down is own lower value
    FFX_DNSR_Shadows_AccumulateMoments(center, FFX_DNSR_Shadows_KernelWeight(0), local_neighborhood);
    FFX_DNSR_Shadows_AccumulateMoments(upper, FFX_DNSR_Shadows_KernelWeight(KERNEL_RADIUS), local_neighborhood);
    FFX_DNSR_Shadows_AccumulateMoments(lower, FFX_DNSR_Shadows_KernelWeight(KERNEL_RADIUS), local_neighborhood);

    // Then read the neighboring values.
    for (int i = 1; i < KERNEL_RADIUS; ++i) {
        float upper_value = g_FFX_DNSR_Shadows_neighborhood[gtid.x][8 + gtid.y - i];
        float lower_value = g_FFX_DNSR_Shadows_neighborhood[gtid.x][8 + gtid.y + i];
        float weight = FFX_DNSR_Shadows_KernelWeight(i);
        FFX_DNSR_Shadows_AccumulateMoments(upper_value, weight, local_neighborhood);
        FFX_DNSR_Shadows_AccumulateMoments(lower_value, weight, local_neighborhood);
    }

    return local_neighborhood;
}

void FFX_DNSR_Shadows_WriteTileMetaData(uint2 gid, uint2 gtid, bool is_cleared, bool all_in_light)
{
    if (all(gtid == 0)) {
        uint light_mask = all_in_light ? TILE_META_DATA_LIGHT_MASK : 0;
        uint clear_mask = is_cleared ? TILE_META_DATA_CLEAR_MASK : 0;
        uint mask = light_mask | clear_mask;
        FFX_DNSR_Shadows_WriteMetadata(gid.y * FFX_DNSR_Shadows_RoundedDivide(FFX_DNSR_Shadows_GetBufferDimensions().x, 8) + gid.x, mask);
    }
}

void FFX_DNSR_Shadows_ClearTargets(uint2 did, uint2 gtid, uint2 gid, float shadow_value, bool is_shadow_receiver, bool all_in_light)
{
    FFX_DNSR_Shadows_WriteTileMetaData(gid, gtid, true, all_in_light);
    FFX_DNSR_Shadows_WriteReprojectionResults(did, float2(shadow_value, 0)); // mean, variance

    float temporal_sample_count = is_shadow_receiver ? 1 : 0;
    FFX_DNSR_Shadows_WriteMoments(did, float3(shadow_value, 0, temporal_sample_count));// mean, variance, temporal sample count
}

void FFX_DNSR_Shadows_TileClassification(uint group_index, uint2 gid)
{
    uint2 gtid = FFX_DNSR_Shadows_RemapLane8x8(group_index); // Make sure we can use the QuadReadAcross intrinsics to access a 2x2 region.
    uint2 did = gid * 8 + gtid;

    bool is_shadow_receiver = FFX_DNSR_Shadows_IsShadowReciever(did);

    bool skip_sky = FFX_DNSR_Shadows_ThreadGroupAllTrue(!is_shadow_receiver);
    if (skip_sky) {
        // We have to set all resources of the tile we skipped to sensible values as neighboring active denoiser tiles might want to read them.
        FFX_DNSR_Shadows_ClearTargets(did, gtid, gid, 0, is_shadow_receiver, false);
        return;
    }

    bool all_in_light = false;
    bool all_in_shadow = false;
    FFX_DNSR_Shadows_SearchSpatialRegion(gid, all_in_light, all_in_shadow);
    float shadow_value = all_in_light ? 1 : 0; // Either all_in_light or all_in_shadow must be true, otherwise we would not skip the tile.

    bool can_skip = all_in_light || all_in_shadow;
    // We have to append the entire tile if there is a single lane that we can't skip
    bool skip_tile = FFX_DNSR_Shadows_ThreadGroupAllTrue(can_skip);
    if (skip_tile) {
        // We have to set all resources of the tile we skipped to sensible values as neighboring active denoiser tiles might want to read them.
        FFX_DNSR_Shadows_ClearTargets(did, gtid, gid, shadow_value, is_shadow_receiver, all_in_light);
        return;
    }

    FFX_DNSR_Shadows_WriteTileMetaData(gid, gtid, false, false);

    float depth = FFX_DNSR_Shadows_ReadDepth(did);
    const float2 velocity = FFX_DNSR_Shadows_GetClosestVelocity(did.xy, depth); // Must happen before we deactivate lanes
    const float local_neighborhood = FFX_DNSR_Shadows_ComputeLocalNeighborhood(did, gtid);

    const float2 texel_size = FFX_DNSR_Shadows_GetInvBufferDimensions();
    const float2 uv = (did.xy + 0.5f) * texel_size;
    const float2 history_uv = uv - velocity;
    const int2 history_pos = history_uv * FFX_DNSR_Shadows_GetBufferDimensions();

    const uint2 tile_index = FFX_DNSR_Shadows_GetTileIndexFromPixelPosition(did);
    const uint linear_tile_index = FFX_DNSR_Shadows_LinearTileIndex(tile_index, FFX_DNSR_Shadows_GetBufferDimensions().x);

    const uint shadow_tile = FFX_DNSR_Shadows_ReadRaytracedShadowMask(linear_tile_index);

    float3 moments_current = 0;
    float variance = 0;
    float shadow_clamped = 0;
    if (is_shadow_receiver) // do not process sky pixels
    {
        bool hit_light = shadow_tile & FFX_DNSR_Shadows_GetBitMaskFromPixelPosition(did);
        const float shadow_current = hit_light ? 1.0 : 0.0;

        // Perform moments and variance calculations
        {
            //bool is_disoccluded = FFX_DNSR_Shadows_IsDisoccluded(did, depth, velocity);
            bool is_disoccluded = true;

            const float3 previous_moments = is_disoccluded ? float3(0.0f, 0.0f, 0.0f) // Can't trust previous moments on disocclusion
                : FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(history_pos);

            const float old_m = previous_moments.x;
            const float old_s = previous_moments.y;
            const float sample_count = previous_moments.z + 1.0f;
            const float new_m = old_m + (shadow_current - old_m) / sample_count;
            const float new_s = old_s + (shadow_current - old_m) * (shadow_current - new_m);

            variance = (sample_count > 1.0f ? new_s / (sample_count - 1.0f) : 1.0f);
            moments_current = float3(new_m, new_s, sample_count);
        }

        // Retrieve local neighborhood and reproject
        {
            float mean = local_neighborhood;
            float spatial_variance = local_neighborhood;

            spatial_variance = max(spatial_variance - mean * mean, 0.0f);

            // Compute the clamping bounding box
            const float std_deviation = sqrt(spatial_variance);
            const float nmin = mean - 0.5f * std_deviation;
            const float nmax = mean + 0.5f * std_deviation;

            // Clamp reprojected sample to local neighborhood
            float shadow_previous = shadow_current;
            if (FFX_DNSR_Shadows_IsFirstFrame() == 0) {
                shadow_previous = FFX_DNSR_Shadows_ReadHistory(history_uv);
            }

            shadow_clamped = clamp(shadow_previous, nmin, nmax);

            // Reduce history weighting
            float const sigma = 20.0f;
            float const temporal_discontinuity = (shadow_previous - mean) / max(0.5f * std_deviation, 0.001f);
            float const sample_counter_damper = exp(-temporal_discontinuity * temporal_discontinuity / sigma);
            moments_current.z *= sample_counter_damper;

            // Boost variance on first frames
            if (moments_current.z < 16.0f) {
                const float variance_boost = max(16.0f - moments_current.z, 1.0f);
                variance = max(variance, spatial_variance);
                variance *= variance_boost;
            }
        }

        // Perform the temporal blend
        const float history_weight = sqrt(max(8.0f - moments_current.z, 0.0f) / 8.0f);
        //const float history_weight = 1.0f;
        shadow_clamped = lerp(shadow_clamped, shadow_current, lerp(0.05f, 1.0f, history_weight));
        //shadow_clamped = shadow_current;
    }

    // Output the results of the temporal pass 
    FFX_DNSR_Shadows_WriteReprojectionResults(did.xy, float2(shadow_clamped, variance));
    FFX_DNSR_Shadows_WriteMoments(did.xy, moments_current);
}

[numthreads(8, 8, 1)]
void main(uint group_index : SV_GroupIndex, uint2 gid : SV_GroupID)
{
    FFX_DNSR_Shadows_TileClassification(group_index, gid);
}
