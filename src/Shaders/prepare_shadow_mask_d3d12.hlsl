/**********************************************************************
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
RTShadowDenoiser

prepare_shadow_mask - reads RT output and creates a packed buffer
of the results for use in the denoiser.

********************************************************************/

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 4

#include "ffx_shadows_dnsr/ffx_denoiser_shadows_util.h"


uint LaneIdToBitShift(uint2 localID)
{
    return localID.y * TILE_SIZE_X + localID.x;
}


bool WaveMaskToBool(uint mask, uint2 localID)
{
    return (1 << LaneIdToBitShift(localID.xy)) & mask;
}

cbuffer PassData : register(b0)
{
    int2 BufferDimensions;
}

Texture2D<uint> t2d_hitMaskResults : register(t0);
RWStructuredBuffer<uint> rwsb_shadowMask : register(u0);

int2 FFX_DNSR_Shadows_GetBufferDimensions()
{
    return BufferDimensions;
}

bool FFX_DNSR_Shadows_HitsLight(uint2 did, uint2 gtid, uint2 gid)
{
    return !WaveMaskToBool(t2d_hitMaskResults[gid], gtid);
}

void FFX_DNSR_Shadows_WriteMask(uint offset, uint value)
{
    rwsb_shadowMask[offset] = value;
}

void FFX_DNSR_Shadows_CopyResult(uint2 gtid, uint2 gid)
{
    const uint2 did = gid * uint2(8, 4) + gtid;
    const uint linear_tile_index = FFX_DNSR_Shadows_LinearTileIndex(gid, FFX_DNSR_Shadows_GetBufferDimensions().x);
    const bool hit_light = FFX_DNSR_Shadows_HitsLight(did, gtid, gid);
    const uint lane_mask = hit_light ? FFX_DNSR_Shadows_GetBitMaskFromPixelPosition(did) : 0;
    FFX_DNSR_Shadows_WriteMask(linear_tile_index, WaveActiveBitOr(lane_mask));
}

void FFX_DNSR_Shadows_PrepareShadowMask(uint2 gtid, uint2 gid)
{
    gid *= 4;
    uint2 tile_dimensions = (FFX_DNSR_Shadows_GetBufferDimensions() + uint2(7, 3)) / uint2(8, 4);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint2 tile_id = uint2(gid.x + i, gid.y + j);
            tile_id = clamp(tile_id, 0, tile_dimensions - 1);
            FFX_DNSR_Shadows_CopyResult(gtid, tile_id);
        }
    }
}

[numthreads(TILE_SIZE_X, TILE_SIZE_Y, 1)]
void main(uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    FFX_DNSR_Shadows_PrepareShadowMask(gtid, gid);
}
