// The following fields must be in user db table 
// Example

// user_id         : 124324154654365465465
// type            : 'Aggressive',
// score_rtq       : rtqScore,
// score_behavioral: behavioralScore,
// score_capacity  : capacityScore,
// score_total     : totalScore



class RiskAssessmentSystem {
    constructor() {
        this.answers = {
            rtq: [],
            behavioral: [],
            capacity: []
        };

        // RTQ Questions - 3 асуулт
        this.rtqQuestions = [
            {
                id: 'rtq1',
                text: "Та хөрөнгө оруулалт хийсэн дүнгээсээ алдагдал хүлээж эхэлвэл ямар мэдрэмж авах вэ?",
                weight: 3,
                options: {
                    A: "Би ямар ч мөнгө алдаж чадахгүй. Би тийм үйл явдалд бэлэн биш бөгөөд маш тухгүй байдалд орно.",
                    B: "Би алдагдал хүлээж эхэлхэд үүнийг арилжааны нэг хэсэг хэмээн хүлээн авч тайван байна.",
                    C: "Үргэлжид тайван байна өндөр эрсдэл өндөр өгөөжтэй байдаг гэж бодно."
                }
            },
            {
                id: 'rtq2',
                text: "Та хөрөнгө оруулалт хийх хугацааг хэр удаан үргэлжилбэл зүгээр гэж боддог вэ?",
                weight: 2,
                options: {
                    A: "Миний мөнгө надад 3 жил дотроо хэрэгтэй.",
                    B: "Би 3-10 жилийн хугацаанд хөрөнгө оруулалт хийнэ.",
                    C: "Би 10-с урт жилийн хугацаанд хөрөнгө оруулалт хийнэ."
                }
            },
            {
                id: 'rtq3',
                text: "Зах зээл маш хүчтэй унавал та ямар үйлдэл хийх вэ?",
                weight: 3,
                options: {
                    A: "Би нэмж алдагдал хүлээхээс сэргийлж бүх авсан хувьцаагаа зарна.",
                    B: "Зах зээл эргэн сэргэхийг хүлээнэ.",
                    C: "Илүү хямд үнээр ахин авах боломж гэж үзэж нэмж авна."
                }
            }
        ];

        // Behavioral Questions - 5 асуулт
        this.behavioralQuestions = [
            {
                id: 'bfa1',
                text: "Та өөрийн хөрөнгө оруулалтын багцаа хэр давтамжтай шалгадаг вэ?",
                weight: 2,
                options: {
                    A: "Өдөр бүр - би зах зээлийн хэмнэлээс хоцрохыг хүсдэггүй.",
                    B: "Сардаа нэг хоёр удаа - Хааяа л нэг шалгадаг.",
                    C: "Маш цөөхөн – Улиралаа, жилдээ цөөхөн шалгадаг."
                }
            },
            {
                id: 'bfa2',
                text: "Хөрөнгө оруулалтын шийдвэр гаргахдаа та ихэвчлэн:",
                weight: 2,
                options: {
                    A: "Маш их судалгаа хийж, бүх мэдээллийг цуглуулсны дараа шийдвэр гаргадаг",
                    B: "Суурь судалгаа хийж, туршлагатай хүмүүсээс зөвлөгөө авдаг",
                    C: "Мэдрэмж, туршлагадаа тулгуурлан шийдвэр гаргадаг"
                }
            },
            {
                id: 'bfa3',
                text: "Таны хөрөнгө оруулалтын тухай ойлголт, мэдлэгийг цөөн үгээр тодорхойлвол?",
                weight: 2,
                options: {
                    A: "Анхан шатны мэдлэгтэй, нарийн зүйлсийг мэдэхгүй.",
                    B: "Суурь ойлголтуудыг мэдхээс гадна өмнө нь хөрөнгө оруулалт хийж байсан.",
                    C: "Маш туршлагатай, зах зээлийн үйл хөдлөлийг идэвхитэй ажиглаж судалдаг."
                }
            },
            {
                id: 'bfa4',
                text: "Хөрөнгө оруулалтын алдагдал хүлээсэн үед та:",
                weight: 2,
                options: {
                    A: "Маш их сэтгэл санаагаар унаж, нойр хүрэхгүй байдаг",
                    B: "Санаа зовдог ч тайван байхыг хичээдэг",
                    C: "Энэ нь зах зээлийн нэг хэсэг гэж хүлээн авдаг"
                }
            },
            {
                id: 'bfa5',
                text: "Таны хөрөнгө оруулалтын зорилго юу вэ?",
                weight: 2,
                options: {
                    A: "Хөрөнгөө хадгалах. Одоо байгаа хөрөнгийн үнэ цэнээ хадгалахыг хүсч байна.",
                    B: "Тэнцвэржүүлсэн өсөлт. Өсөлттэй хувьцаа болон тогтмол орлого хослуулан багц бүрдүүлж хөрөнгөө өсгөх.",
                    C: "Хөрөнгөө хэд дахин өсгөж үржүүлэх: Хамгийн өндөр өгөөжтэй хувьцаануудад хөрөнгө оруулалт хийх."
                }
            }
        ];

        // Capacity Questions - 3 асуулт
        this.capacityQuestions = [
            {
                id: 'rcs1',
                text: "Таны одоогийн сарын орлого:",
                weight: 3,
                options: {
                    A: "Зарлагаа хасаад бага зэрэг үлддэг",
                    B: "Зарлагаа хасаад тодорхой хэмжээний мөнгө хадгалж чаддаг",
                    C: "Зарлагаа хасаад их хэмжээний мөнгө хадгалж чаддаг"
                }
            },
            {
                id: 'rcs2',
                text: "Таны санхүүгийн үүрэг хариуцлага (зээл, өр төлбөр):",
                weight: 3,
                options: {
                    A: "Их хэмжээний зээл, өртэй",
                    B: "Дунд зэргийн зээлтэй",
                    C: "Зээл, өр төлбөргүй эсвэл маш бага"
                }
            },
            {
                id: 'rcs3',
                text: "Таны онцгой үеийн нөөц (emergency fund):",
                weight: 2,
                options: {
                    A: "1 сарын зарлагаас бага",
                    B: "3-6 сарын зарлагатай тэнцэх",
                    C: "6-с дээш сарын зарлагатай тэнцэх"
                }
            }
        ];
    }

    addAnswer(category, questionId, answer) {
        if (category === 'rtq') {
            this.answers.rtq.push({ id: questionId, answer });
        } else if (category === 'behavioral') {
            this.answers.behavioral.push({ id: questionId, answer });
        } else if (category === 'capacity') {
            this.answers.capacity.push({ id: questionId, answer });
        }
    }

    calculateScore(answers, questions) {
        let totalScore = 0;
        let totalWeight = 0;

        answers.forEach(ans => {
            const question = questions.find(q => q.id === ans.id);
            if (question) {
                const optionScore = ans.answer === 'A' ? 1 : ans.answer === 'B' ? 2 : 3;
                totalScore += optionScore * question.weight;
                totalWeight += question.weight;
            }
        });

        return totalScore / totalWeight;
    }

    calculateRiskProfile() {
        const rtqScore = this.calculateScore(this.answers.rtq, this.rtqQuestions);
        const behavioralScore = this.calculateScore(this.answers.behavioral, this.behavioralQuestions);
        const capacityScore = this.calculateScore(this.answers.capacity, this.capacityQuestions);

        // Дундаж оноо
        const totalScore = (rtqScore * 0.35 + behavioralScore * 0.35 + capacityScore * 0.3);

        // Профайл тодорхойлох логик
        if (totalScore <= 1.75) {
            return {
                type: 'Conservative',
                riskLevel: 'Бага',
                description: 'Та эрсдэл хүлээн авах дургүй бөгөөд хөрөнгөө найдвартай байршуулахыг эрхэмлэдэг.',
                recommendations: [
                    'Хөрөнгө оруулагчийн эрсдэл хүлээж авах чадварын тест өгсөнд баярлалаа. Бид таны эрсдэл хүлээн авах чадварт тохируулан сонирхох боломжтой хувьцаануудыг тооцоолж, мэдээ мэдээллээр цаг алдалгүй хангаж байх болно.'
                ],
                scores: {
                    rtq: rtqScore,
                    behavioral: behavioralScore,
                    capacity: capacityScore,
                    total: totalScore
                }
            };
        } else if (totalScore <= 2.25) {
            return {
                type: 'Moderate',
                riskLevel: 'Дунд',
                description: 'Та эрсдэл, өгөөжийг тэнцвэртэй харж үздэг.',
                recommendations: [
                    'Хөрөнгө оруулагчийн эрсдэл хүлээж авах чадварын тест өгсөнд баярлалаа. Бид таны эрсдэл хүлээн авах чадварт тохируулан сонирхох боломжтой хувьцаануудыг тооцоолж, мэдээ мэдээллээр цаг алдалгүй хангаж байх болно.'
                ],
                scores: {
                    rtq: rtqScore,
                    behavioral: behavioralScore,
                    capacity: capacityScore,
                    total: totalScore
                }
            };
        } else {
            return {
                type: 'Aggressive',
                riskLevel: 'Өндөр',
                description: 'Та өндөр өгөөжийн төлөө эрсдэл хүлээх бэлэн байдаг.',
                recommendations: [
                    'Хөрөнгө оруулагчийн эрсдэл хүлээж авах чадварын тест өгсөнд баярлалаа. Бид таны эрсдэл хүлээн авах чадварт тохируулан сонирхох боломжтой хувьцаануудыг тооцоолж, мэдээ мэдээллээр цаг алдалгүй хангаж байх болно.'
                ],
                scores: {
                    rtq: rtqScore,
                    behavioral: behavioralScore,
                    capacity: capacityScore,
                    total: totalScore
                }
            };
        }
    }

    validateAnswers() {
        return this.answers.rtq.length === this.rtqQuestions.length &&
               this.answers.behavioral.length === this.behavioralQuestions.length &&
               this.answers.capacity.length === this.capacityQuestions.length;
    }

    getRTQQuestions() {
        return this.rtqQuestions;
    }

    getBehavioralQuestions() {
        return this.behavioralQuestions;
    }

    getCapacityQuestions() {
        return this.capacityQuestions;
    }
}

// Хэрэглээний жишээ
const assessment = new RiskAssessmentSystem();
assessment.addAnswer('rtq', 'rtq1', 'B');
assessment.addAnswer('rtq', 'rtq2', 'B');
assessment.addAnswer('rtq', 'rtq3', 'B');
assessment.addAnswer('behavioral', 'bfa1', 'B');
assessment.addAnswer('behavioral', 'bfa2', 'B');
assessment.addAnswer('behavioral', 'bfa3', 'B');
assessment.addAnswer('behavioral', 'bfa4', 'B');
assessment.addAnswer('behavioral', 'bfa5', 'C');
assessment.addAnswer('capacity', 'rcs1', 'B');
assessment.addAnswer('capacity', 'rcs2', 'B');
assessment.addAnswer('capacity', 'rcs3', 'B');

if (assessment.validateAnswers()) {
    const profile = assessment.calculateRiskProfile();
    console.log('Risk Profile:', profile);
} else {
    console.log('Please answer all questions correctly.');
}


