import { motion } from "framer-motion";
import { Plane, Calendar, Clock, Armchair, ChevronRight } from "lucide-react";

export interface Flight {
  flight_id: string;
  airline?: string;
  origin?: string;
  destination?: string;
  date?: string;
  departure_time?: string;
  cabin?: string;
  price?: number;
  available_seats?: number;
}

interface Props {
  flights: Flight[];
  onBook: (flightId: string) => void;
}

export function FlightResultCard({ flights, onBook }: Props) {
  if (!flights || flights.length === 0) return null;

  return (
    <div className="mt-3 flex w-full flex-col gap-2">
      {flights.map((flight) => (
        <motion.div
          key={flight.flight_id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col gap-3 rounded-xl border border-indigo-400/20 bg-indigo-500/[0.04] p-4 backdrop-blur-md transition hover:border-indigo-400/40 hover:bg-indigo-500/[0.08]"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-indigo-500/20 text-indigo-300">
                <Plane className="h-4 w-4" />
              </span>
              <div>
                <div className="font-semibold text-white/95">
                  {flight.origin ?? "Unknown"} <span className="text-white/40 mx-1">→</span> {flight.destination ?? "Unknown"}
                </div>
                <div className="text-[11px] font-medium text-white/50 uppercase tracking-widest">
                  {flight.airline ?? "Airline"} • {flight.flight_id}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-white/95">${flight.price ?? "—"}</div>
              <div className="text-[11px] text-white/50">{flight.available_seats ?? 0} seats left</div>
            </div>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-2 rounded-lg bg-white/[0.02] p-2.5">
            <div className="flex items-center gap-1.5 text-[12px] text-white/70">
              <Calendar className="h-3.5 w-3.5 text-white/40" />
              <span>{flight.date ?? "Any date"}</span>
            </div>
            <div className="flex items-center gap-1.5 text-[12px] text-white/70">
              <Clock className="h-3.5 w-3.5 text-white/40" />
              <span>{flight.departure_time ?? "Any time"}</span>
            </div>
            <div className="flex items-center gap-1.5 text-[12px] text-white/70">
              <Armchair className="h-3.5 w-3.5 text-white/40" />
              <span className="capitalize">{flight.cabin ?? "Economy"}</span>
            </div>
          </div>

          <button
            onClick={() => onBook(flight.flight_id)}
            className="group mt-1 flex w-full items-center justify-center gap-2 rounded-lg bg-white/[0.05] py-2 text-[12px] font-semibold text-white/90 transition hover:bg-indigo-500 hover:text-white"
          >
            Book this flight
            <ChevronRight className="h-3.5 w-3.5 opacity-50 transition group-hover:translate-x-1 group-hover:opacity-100" />
          </button>
        </motion.div>
      ))}
    </div>
  );
}
