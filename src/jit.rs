use capstone::Capstone;
use memmap2::{Mmap, MmapMut};

pub struct Jit {
    buffer: Vec<u8>,
}

impl Jit {
    pub fn ret(&mut self) {
        self.buffer.push(0xc3);
    }

    pub fn imm32(&mut self, imm: i32) {
        let bytes: [u8; 4] = imm.to_le_bytes();
        self.buffer.push(bytes[0]);
        self.buffer.push(bytes[1]);
        self.buffer.push(bytes[2]);
        self.buffer.push(bytes[3]);
    }

    pub fn movri_64(&mut self, target: Rq, imm: i32) {
        // MOV r/m64, imm32  --  REX.W + C7 /0 id
        let m = modrm(ModRm::Register, 0, (target as u8) & 0x7);
        let r = rex(1, 0, 0, (target as u8) >> 3);
        self.buffer.push(r);
        self.buffer.push(0xc7);
        self.buffer.push(m);
        self.imm32(imm);
    }

    pub fn build(self) -> Mmap {
        let mut mm = MmapMut::map_anon(self.buffer.len()).unwrap();
        mm.copy_from_slice(&self.buffer);
        mm.make_exec().unwrap()
    }
}

type JitFun = unsafe extern fn() -> u64;




pub fn try_it() {
    let mut jit = Jit { buffer: Vec::new() };

    jit.movri_64(Rq::RAX, 42);
    jit.ret();

    disasm(&jit.buffer);

    let mm = jit.build();
    println!("{:?}", mm);

    println!("f = {:?}", mm.as_ptr() );
    let f: JitFun = unsafe { std::mem::transmute(mm.as_ptr()) };
    
    println!("result: {}", unsafe { f() })
}

pub fn disasm(code: &[u8]) {
    use capstone::prelude::*;
    let cs = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .syntax(arch::x86::ArchSyntax::Intel)
        .detail(true)
        .build()
        .expect("Failed to create Capstone object");

    let insns = cs.disasm_all(code, 0x1000).expect("Failed to disassemble");

    for i in insns.as_ref() {
        println!("{}       ; {:x?}", i, i.bytes());
    }
    // println!("Found {} instructions", insns.len());
    // for i in insns.as_ref() {
    //     println!();
    //     println!("{}", i);

    //     let detail: InsnDetail = cs.insn_detail(&i).expect("Failed to get insn detail");
    //     let arch_detail: ArchDetail = detail.arch_detail();
    //     let ops = arch_detail.operands();

    //     let output: &[(&str, String)] = &[
    //         ("insn id:", format!("{:?}", i.id().0)),
    //         ("bytes:", format!("{:?}", i.bytes())),
    //         ("read regs:", reg_names(&cs, detail.regs_read())),
    //         ("write regs:", reg_names(&cs, detail.regs_write())),
    //         ("insn groups:", group_names(&cs, detail.groups())),
    //     ];

    //     for &(ref name, ref message) in output.iter() {
    //         println!("{:4}{:12} {}", "", name, message);
    //     }

    //     println!("{:4}operands: {}", "", ops.len());
    //     for op in ops {
    //         println!("{:8}{:?}", "", op);
    //     }
    // }
}

/// Print register names
fn reg_names(cs: &capstone::Capstone, regs: &[capstone::RegId]) -> String {
    let names: Vec<String> = regs.iter().map(|&x| cs.reg_name(x).unwrap()).collect();
    names.join(", ")
}

/// Print instruction group names
fn group_names(cs: &Capstone, regs: &[capstone::InsnGroupId]) -> String {
    let names: Vec<String> = regs.iter().map(|&x| cs.group_name(x).unwrap()).collect();
    names.join(", ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rq {
    RAX = 0x0,
    RCX = 0x1,
    RDX = 0x2,
    RBX = 0x3,
    RSP = 0x4,
    RBP = 0x5,
    RSI = 0x6,
    RDI = 0x7,
    R8 = 0x8,
    R9 = 0x9,
    R10 = 0xA,
    R11 = 0xB,
    R12 = 0xC,
    R13 = 0xD,
    R14 = 0xE,
    R15 = 0xF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rx {
    XMM0 = 0x0,
    XMM1 = 0x1,
    XMM2 = 0x2,
    XMM3 = 0x3,
    XMM4 = 0x4,
    XMM5 = 0x5,
    XMM6 = 0x6,
    XMM7 = 0x7,
    XMM8 = 0x8,
    XMM9 = 0x9,
    XMM10 = 0xA,
    XMM11 = 0xB,
    XMM12 = 0xC,
    XMM13 = 0xD,
    XMM14 = 0xE,
    XMM15 = 0xF,
}

pub fn rex(w: u8, r: u8, x: u8, b: u8) -> u8 {
    debug_assert!(w <= 1 && r <= 1 && x <= 1 && b <= 1);
    0x40 | (w << 3) | (r << 2) | (x << 1) | b
}

pub fn modrm(mo: ModRm, reg: u8, rm: u8) -> u8 {
    debug_assert!(reg <= 7 && rm <= 7);
    ((mo as u8) << 6) | (reg << 3) | rm
}

pub enum ModRm {
    Address = 0x0,
    Offset8 = 0x1,
    Offset32 = 0x2,
    Register = 0x3,
}
