// The values of the renderer API (`docs/api.md` section 2) and the
// payloads of its events (section 8), as TypeScript types.

/** A frame id: a whole number from 0 to below 2^53. */
export type FrameId = number;

/** Three components along a frame's or the camera's axes. */
export type Vec3 = [number, number, number];

/** A camera pose (section 2): position in meters, orientation a unit quaternion `[x, y, z, w]`, speed in m/s. */
/** Options of `setCamera`. */
export interface SetCameraOptions {
    /** The camera moves and turns with its frame (default false: free). */
    ride?: boolean;
}

export interface Pose {
    /** The frame the camera is in. */
    frameId: FrameId;
    /** Meters from the frame origin, in the frame's axes. */
    position: Vec3;
    /** The camera axes relative to the frame axes, `[x, y, z, w]`. */
    orientation: [number, number, number, number];
    /** The set flight speed, m/s. */
    speed: number;
}

/** The hub to connect to (section 3); `buildId` may be empty. */
export interface Connection {
    /** The hub's base URL. */
    hubUrl: string;
    /** The key sent with every request. */
    apiKey: string;
    /** The space to draw. */
    spaceId: string;
    /** The build to draw; empty for the space's current one. */
    buildId?: string;
}

/** The view and time options by camelCase name (section 3), parsed by the renderer's options parser. */
export type ViewOptions = Record<string, string | number | boolean | undefined>;

/** Who flies the camera from the keyboard and mouse (section 4). */
export type Controls = "builtin" | "none";

/** The free camera's held input (section 4). */
export interface FlightInput {
    /** Components in -1 to 1 along the camera axes. */
    move: Vec3;
    /** Radians to turn this tick: yaw, pitch, roll. */
    look: Vec3;
    /** Boost held or not. */
    boost: boolean;
}

/** The time drawn (section 5), UTC milliseconds. */
export interface TimeState {
    /** The instant drawn. */
    utcMs: number;
    /** The universe's origin. */
    originUtcMs: number;
    /** Real now. */
    nowUtcMs: number;
    /** The playback rate. */
    timeScale: number;
}

/** A frame's surface shape from its level-0 tiles (section 5), meters. */
export interface SurfaceShape {
    /** Equatorial radius of the reference ellipsoid. */
    radiusEquatorial: number;
    /** Polar radius of the reference ellipsoid. */
    radiusPolar: number;
    /** The lowest height a tile may hold. */
    heightFloor: number;
}

/** One frame of the registry (section 5). */
export interface FrameInfo {
    /** The frame's id. */
    id: FrameId;
    /** The parent frame's id, null for the root. */
    parentId: FrameId | null;
    /** Kilograms. */
    mass: number;
    /** Radians per second, in the frame's axes. */
    angularVelocity: Vec3;
    /** Whether the frame has surface tiles. */
    hasSurface: boolean;
    /** The surface shape once level-0 tiles arrived, else null. */
    surface: SurfaceShape | null;
}

/** Whether `transform` moves a point or a direction (section 5). */
export type TransformKind = "point" | "direction";

/** The drawn surface height along a direction (section 6). */
export interface SurfaceHeight {
    /** Meters above the reference ellipsoid. */
    height: number;
    /** The tile level it came from. */
    level: number;
    /** Whether a finer tile there is still on its way. */
    finer: boolean;
}

/** What a pixel hits (section 6). */
export interface RaycastHit {
    /** The frame hit. */
    frameId: FrameId;
    /** The point hit, in the frame's axes. */
    position: Vec3;
    /** Meters from the camera. */
    distance: number;
    /** What was hit. */
    kind: "surface" | "cells";
}

/** A point projected into the view (section 6), CSS pixels. */
export interface Projection {
    /** Pixels from the left. */
    x: number;
    /** Pixels from the top. */
    y: number;
    /** Depth of the point. */
    depth: number;
}

/** One layer of the build (section 7). */
export interface LayerInfo {
    /** The layer's id. */
    id: string;
    /** Whether the layer is an overlay. */
    overlay: boolean;
    /** Whether showing it hides the other exclusive layers. */
    exclusive: boolean;
    /** Whether it is shown as currently applied. */
    shown: boolean;
}

/** The exposure (section 7). */
export type Exposure = { mode: "auto"; biasStops: number } | { mode: "fixed"; stops: number };

/** How surfaces are drawn (section 7). */
export type MeshMode = "surfaces" | "vertices" | "edges";

/** The `frameChanged` event (section 8). */
export interface FrameChange {
    /** The frame the camera left. */
    from: FrameId;
    /** The frame the camera is in now. */
    to: FrameId;
}

/** The `settled` event (section 8). */
export interface Settled {
    /** Seconds until everything the view asked for was drawn. */
    seconds: number;
}

/** The `hit` event (section 8): a raycast result or null, with the click. */
export interface Hit {
    /** What the click hit, or null. */
    hit: RaycastHit | null;
    /** CSS pixels from the left. */
    x: number;
    /** CSS pixels from the top. */
    y: number;
    /** The mouse button. */
    button: number;
}

/** The `stats` event (section 8), once a second. */
export interface Stats {
    /** Frames per second, 0 where nothing is drawn. */
    fps: number;
    /** Surface tiles drawn. */
    tilesDrawn: number;
    /** Triangles drawn. */
    triangles: number;
    /** Cells drawn. */
    cellsDrawn: number;
    /** Cells still waiting for sections. */
    streaming: number;
    /** Module runs so far this session. */
    computing: number;
    /** Offers waiting or being sent. */
    uploading: number;
}

/** What each event (section 8) carries. */
export interface RendererEvents {
    /** The loop runs. */
    ready: undefined;
    /** Something stopped the loop or a queued call failed. */
    error: string;
    /** The pose changed this tick. */
    camera: Pose;
    /** The camera's frame changed. */
    frameChanged: FrameChange;
    /** Everything the view asked for is drawn. */
    settled: Settled;
    /** A click. */
    hit: Hit;
    /** Once a second. */
    stats: Stats;
}

/** The names of the events (section 8). */
export type RendererEvent = keyof RendererEvents;
