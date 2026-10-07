//! Bit-exact port of `numpy.random.default_rng(seed).integers(low, high, n)` (SeedSequence ->
//! PCG64 -> buffered Lemire for 32-bit ranges), used only for the front end's warm-up noise so the
//! first 16 frames after a reset score exactly like the Python reference.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL: usize = 4;

/// `SeedSequence(seed).generate_state(4, np.uint64)` for a seed below 2^32.
fn seed_state(seed: u32) -> [u64; 4] {
    let mut hc = INIT_A;
    let mut hashmix = |v: u32| {
        let mut v = v ^ hc;
        hc = hc.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hc);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let entropy = [seed];
    let mut pool = [0u32; POOL];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    let mut hb = INIT_B;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % POOL] ^ hb;
        hb = hb.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hb);
        *w = v ^ (v >> 16);
    }
    std::array::from_fn(|i| words[2 * i] as u64 | ((words[2 * i + 1] as u64) << 32))
}

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

struct Pcg64 {
    state: u128,
    inc: u128,
    buffered: Option<u32>,
}

impl Pcg64 {
    fn new(seed: u32) -> Self {
        let s = seed_state(seed);
        let initstate = ((s[0] as u128) << 64) | s[1] as u128;
        let initseq = ((s[2] as u128) << 64) | s[3] as u128;
        let mut r = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            buffered: None,
        };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((hi >> 58) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.buffered.take() {
            return v;
        }
        let n = self.next_u64();
        self.buffered = Some((n >> 32) as u32);
        n as u32
    }
}

/// `numpy.random.default_rng(seed).integers(low, high, n)` (high exclusive, range < 2^32).
pub fn numpy_integers(seed: u32, low: i64, high: i64, n: usize) -> Vec<i64> {
    let mut g = Pcg64::new(seed);
    let rng = (high - 1 - low) as u32;
    let excl = rng.wrapping_add(1);
    (0..n)
        .map(|_| {
            let mut m = g.next_u32() as u64 * excl as u64;
            let mut left = m as u32;
            if left < excl {
                let threshold = (u32::MAX - rng) % excl;
                while left < threshold {
                    m = g.next_u32() as u64 * excl as u64;
                    left = m as u32;
                }
            }
            low + (m >> 32) as i64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy() {
        // numpy 2.5.3: SeedSequence(0).generate_state(4, np.uint64), default_rng(0).integers(-1000, 1000, 64000)
        assert_eq!(
            seed_state(0),
            [
                0xdb2cd7e7b0f478be,
                0xabf4641a2c71ba49,
                0x20c6ed6d9d7b8d41,
                0x2c4099de223c39d4
            ]
        );
        let w = numpy_integers(0, -1000, 1000, 64000);
        assert_eq!(&w[..8], &[701, 273, 22, -461, -385, -919, -850, -967]);
        assert_eq!(&w[w.len() - 4..], &[456, -166, 442, -14]);
        assert_eq!(w.iter().sum::<i64>(), -134585);
    }
}
